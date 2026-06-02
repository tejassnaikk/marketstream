"""
verification/orderbook_reconstructor.py

Reconstructs a local BTC/USDT order book from the Binance US depth stream,
applying incremental diffs consumed from Kafka on top of a REST snapshot.

The core correctness guarantee rests on two invariants:
  1. The first applied diff must overlap the snapshot's lastUpdateId.
  2. Every subsequent diff must immediately follow the previous one (no gaps).

Violating either invariant produces a silently corrupt book. ResyncRequired is
raised so the caller can rebuild from a fresh snapshot rather than silently
accumulate errors.
"""

import json
import logging
import ssl
import time
import urllib.request
from datetime import datetime, timezone

import certifi
from kafka import KafkaConsumer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SNAPSHOT_URL = "https://api.binance.us/api/v3/depth?symbol=BTCUSDT&limit=100"
KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPIC = "btcusdt_depth"


class ResyncRequired(Exception):
    """
    Raised when the local order book has diverged from the exchange and cannot
    be recovered by applying further stream diffs. The caller must re-fetch a
    REST snapshot and restart the application sequence.
    """


class OrderBookReconstructor:

    def __init__(self) -> None:
        # Price strings are used as keys intentionally. Float keys would silently
        # coerce "71000.00000000" and "71000.0" to the same bucket, hiding data.
        # String equality matches Binance's own key semantics exactly.
        self.bids: dict[str, str] = {}
        self.asks: dict[str, str] = {}
        self.last_update_id: int = 0
        self.initialized: bool = False
        self.buffered_events: list = []

        # Tracks whether the stream has been anchored to the current snapshot.
        # False after every fetch_snapshot() call — the next processed event
        # must pass the overlap anchor check before strict continuity is enforced.
        self._anchored: bool = False

    # -------------------------------------------------------------------------
    # Snapshot
    # -------------------------------------------------------------------------

    def fetch_snapshot(self) -> None:
        """
        Fetch a REST depth snapshot and use it as the book's consistent base state.

        The snapshot must be fetched after the Kafka consumer is running so that
        buffered stream events span the snapshot's lastUpdateId. If the snapshot
        is fetched before any events are buffered, the anchor check on the first
        event may fail because the stream has already moved past the snapshot's
        sequence number, triggering an immediate resync.
        """
        # macOS Python does not use the system certificate store, so without an
        # explicit certifi CA bundle every TLS connection to Binance fails with
        # CERTIFICATE_VERIFY_FAILED. This applies to both the initial fetch and
        # every subsequent verify_against_snapshot() call.
        ctx = ssl.create_default_context(cafile=certifi.where())

        with urllib.request.urlopen(SNAPSHOT_URL, context=ctx) as resp:
            snapshot = json.loads(resp.read())

        self.last_update_id = snapshot["lastUpdateId"]
        self.bids = {price: qty for price, qty in snapshot["bids"]}
        self.asks = {price: qty for price, qty in snapshot["asks"]}
        self.initialized = True

        # Reset anchoring. After a resync the first applied diff must again
        # satisfy U <= lastUpdateId+1 <= u before we trust the sequence to be
        # continuous. Skipping this reset would let a post-resync event bypass
        # the anchor check and potentially apply diffs that don't connect to
        # the new snapshot's sequence position.
        self._anchored = False

        log.info(
            "Snapshot fetched. lastUpdateId=%d, bids=%d, asks=%d",
            self.last_update_id,
            len(self.bids),
            len(self.asks),
        )

    # -------------------------------------------------------------------------
    # Stream processing
    # -------------------------------------------------------------------------

    def process_update(self, event: dict) -> bool:
        """
        Apply one depth update event to the local book.

        Returns True if the event was applied, False if it was buffered or dropped.
        Raises ResyncRequired if a sequencing invariant is violated.
        """

        # Pre-initialization buffer. Events that arrive before fetch_snapshot() is
        # called are saved rather than discarded. The Binance sync algorithm requires
        # a buffered event to span the snapshot's lastUpdateId for the anchor check.
        # Without this, there is a race window between consumer start and snapshot
        # fetch where the anchor event is lost and resync loops indefinitely.
        if not self.initialized:
            self.buffered_events.append(event)
            return False

        # Stale-event drop. The snapshot already incorporates all book changes up to
        # and including lastUpdateId. An event with u <= lastUpdateId has nothing new
        # to contribute; applying it would double-count its changes.
        if event["u"] <= self.last_update_id:
            return False

        if not self._anchored:
            # Anchor check for the first event applied after a snapshot.
            #
            # The Binance synchronization algorithm requires the first applied event
            # to satisfy:   U <= lastUpdateId + 1 <= u
            #
            # This guarantees no gap between the snapshot state and the first diff.
            # If U > lastUpdateId + 1, the stream has already moved past the snapshot
            # and there is an unresolvable hole. If u < lastUpdateId + 1, the event
            # is entirely stale (which the drop check above should have caught).
            if not (event["U"] <= self.last_update_id + 1 <= event["u"]):
                raise ResyncRequired(
                    f"First event anchor failed: "
                    f"U={event['U']}, lastUpdateId+1={self.last_update_id + 1}, u={event['u']}"
                )
            self._anchored = True

        else:
            # Continuity check for every event after the first.
            #
            # event['U'] must be exactly last_update_id + 1. Any larger value means
            # at least one diff was dropped in transit — the local book is now missing
            # one or more price-level changes and cannot be repaired by applying future
            # events. Only a fresh snapshot can restore correctness.
            if event["U"] != self.last_update_id + 1:
                raise ResyncRequired(
                    f"Gap detected: expected U={self.last_update_id + 1}, got {event['U']}"
                )

        # Apply bid changes.
        #
        # qty == "0.00000000" is Binance's deletion signal: no resting orders remain
        # at this price level. The level must be removed from the book entirely.
        # Storing a zero-quantity level would corrupt depth and mid-price calculations.
        # pop() with a default is used so deletion of a price we don't have is a no-op
        # rather than a KeyError — gaps in the initial snapshot make this possible.
        for price, qty in event["b"]:
            if qty == "0.00000000":
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty

        for price, qty in event["a"]:
            if qty == "0.00000000":
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

        # Advance the cursor using event['u'] (the last ID in the batch), not event['U'].
        # This means the next event's U must equal u+1, enforcing strict ordering even
        # when a single message batches many update IDs (as is common under heavy load).
        self.last_update_id = event["u"]
        return True

    # -------------------------------------------------------------------------
    # Read-only views
    # -------------------------------------------------------------------------

    def get_top_levels(self, n: int = 10) -> dict:
        """
        Return the top n price levels on each side plus the best-price spread.

        Sorting on every call is O(k log k) in the full book size. For a verification
        script this is fine; a production system would maintain a sorted structure
        (e.g. a skip list or SortedDict) so that top-of-book reads are O(1).
        """
        sorted_bids = sorted(self.bids.items(), key=lambda x: float(x[0]), reverse=True)[:n]
        sorted_asks = sorted(self.asks.items(), key=lambda x: float(x[0]))[:n]

        best_bid = float(sorted_bids[0][0]) if sorted_bids else None
        best_ask = float(sorted_asks[0][0]) if sorted_asks else None
        spread = round(best_ask - best_bid, 2) if best_bid and best_ask else None

        return {
            "bids": [{"price": p, "qty": q} for p, q in sorted_bids],
            "asks": [{"price": p, "qty": q} for p, q in sorted_asks],
            "spread": spread,
            "last_update_id": self.last_update_id,
        }

    def verify_against_snapshot(self) -> dict:
        """
        Spot-check the reconstructed book against a fresh REST snapshot.

        Compares the top 5 bid prices and top 5 ask prices (10 levels total) between
        the local book and the snapshot. Quantities are intentionally excluded from
        the comparison: the two snapshots are captured at different instants so
        quantity divergence is expected and not indicative of a reconstruction error.
        Only a price-level mismatch (a level present in one book but not the other)
        signals a potential bug in the reconstruction logic.
        """
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(SNAPSHOT_URL, context=ctx) as resp:
            snap = json.loads(resp.read())

        n = 5
        my_bids  = sorted(self.bids.keys(),  key=float, reverse=True)[:n]
        my_asks  = sorted(self.asks.keys(),  key=float)[:n]
        # REST snapshot returns bids best-first (descending) and asks best-first
        # (ascending) already, so slicing [:n] gives the top n without re-sorting.
        snap_bids = [p for p, _ in snap["bids"][:n]]
        snap_asks = [p for p, _ in snap["asks"][:n]]

        mismatches: list[dict] = []
        matched = 0

        for rank, (mine, theirs) in enumerate(zip(my_bids, snap_bids), start=1):
            if mine == theirs:
                matched += 1
            else:
                mismatches.append(
                    {"side": "bid", "rank": rank, "book_price": mine, "snapshot_price": theirs}
                )

        for rank, (mine, theirs) in enumerate(zip(my_asks, snap_asks), start=1):
            if mine == theirs:
                matched += 1
            else:
                mismatches.append(
                    {"side": "ask", "rank": rank, "book_price": mine, "snapshot_price": theirs}
                )

        total = n * 2  # 5 bids + 5 asks
        return {
            "matched": matched,
            "total": total,
            "pct": round(matched / total * 100, 1),
            "mismatches": mismatches,
            "book_last_update_id": self.last_update_id,
            "snap_last_update_id": snap["lastUpdateId"],
        }


# =============================================================================
# Entry point
# =============================================================================

def _print_top(top: dict) -> None:
    ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
    spread_str = f"  spread={top['spread']}" if top["spread"] is not None else ""
    print(f"\n[{ts} UTC]  last_update_id={top['last_update_id']}{spread_str}")
    print("  Bids (best → worst):")
    for lvl in top["bids"]:
        print(f"    {float(lvl['price']):>13,.2f}    {float(lvl['qty']):.8f} BTC")
    print("  Asks (best → worst):")
    for lvl in top["asks"]:
        print(f"    {float(lvl['price']):>13,.2f}    {float(lvl['qty']):.8f} BTC")


def _print_verify(result: dict) -> None:
    log.info(
        "Verification: %d/%d price levels matched (%.1f%%)  "
        "book_id=%d  snap_id=%d",
        result["matched"],
        result["total"],
        result["pct"],
        result["book_last_update_id"],
        result["snap_last_update_id"],
    )
    for mm in result["mismatches"]:
        log.warning(
            "  Mismatch — %s rank %d: book=%s  snapshot=%s",
            mm["side"],
            mm["rank"],
            mm["book_price"],
            mm["snapshot_price"],
        )


if __name__ == "__main__":
    reconstructor = OrderBookReconstructor()

    # Fetch the snapshot before starting the consumer. Any events consumed after
    # fetch_snapshot() will be processed live. If the caller had connected to Kafka
    # first, pre-snapshot events would be buffered and drained below.
    reconstructor.fetch_snapshot()

    # Drain buffered events. In this script the buffer is typically empty because
    # fetch_snapshot() is called before the consumer starts. It is drained anyway
    # because the class may be reused in a context where the consumer starts first.
    drained = 0
    for event in reconstructor.buffered_events:
        try:
            reconstructor.process_update(event)
            drained += 1
        except ResyncRequired as exc:
            log.warning("ResyncRequired while draining buffer: %s — re-fetching snapshot", exc)
            reconstructor.fetch_snapshot()

    reconstructor.buffered_events.clear()
    if drained:
        log.info("Drained %d buffered event(s).", drained)

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        auto_offset_reset="latest",
        enable_auto_commit=False,
        # consumer_timeout_ms is set as a safety valve for the iterator interface.
        # The main loop uses consumer.poll() so this value does not gate message
        # delivery — it prevents the consumer thread from blocking indefinitely if
        # the broker becomes unreachable while using the iterator form elsewhere.
        consumer_timeout_ms=10_000,
        value_deserializer=lambda raw: json.loads(raw.decode("utf-8")),
    )

    log.info("Consuming %s from %s", KAFKA_TOPIC, KAFKA_BOOTSTRAP)

    last_top_print = 0.0
    last_verify_print = 0.0

    try:
        while True:
            # poll() rather than the for-record-in-consumer iterator so that the
            # timing checks below fire even during quiet market periods when no
            # messages arrive. consumer_timeout_ms would kill the iterator after
            # 10 seconds of silence; poll() returns an empty dict and lets us
            # continue the loop without breaking the consumer session.
            batch = consumer.poll(timeout_ms=1_000)

            for _tp, records in batch.items():
                for record in records:
                    try:
                        reconstructor.process_update(record.value)
                    except ResyncRequired as exc:
                        log.error(
                            "ResyncRequired: %s — fetching fresh snapshot and re-anchoring",
                            exc,
                        )
                        # Re-fetch resets _anchored = False so the next processed
                        # event must satisfy the anchor check before strict continuity
                        # is enforced again.
                        reconstructor.fetch_snapshot()

            now = time.monotonic()

            if now - last_top_print >= 5.0:
                _print_top(reconstructor.get_top_levels(5))
                last_top_print = now

            if now - last_verify_print >= 60.0:
                log.info("Running spot-check against REST snapshot…")
                _print_verify(reconstructor.verify_against_snapshot())
                last_verify_print = now

    except KeyboardInterrupt:
        log.info("Interrupted — shutting down.")
    finally:
        consumer.close()
        log.info("Consumer closed.")
