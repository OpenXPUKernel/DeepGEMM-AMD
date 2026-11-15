"""Resident task coverage/ordering oracle; not a device memory-order proof."""

import random
import unittest
from pathlib import Path as Path


class ResidentScheduleTests(unittest.TestCase):
    def test_batched_return_pointers_preserve_all_valid_rows_and_paired_columns(self):
        for columns, stride in ((256, 1), (512, 6)):
            row_stride = 256 // (columns // 8)
            expected_cols = {
                c
                for panel in (0, stride)[: columns // 256]
                for c in range(panel * 256, panel * 256 + 256)
            }
            for batch in (2, 4):
                for valid in (0, 1, 31, 63, 127, 128):
                    seen = set()
                    for base in range(0, 128 // row_stride, batch):
                        for tid in range(256):
                            row0, col = (
                                tid // (columns // 8),
                                tid % (columns // 8) * 8,
                            )
                            logical = col % 256 + col // 256 * stride * 256
                            for j in range(batch):
                                row = row0 + (base + j) * row_stride
                                if row >= valid:
                                    continue
                                for q in range(8):
                                    self.assertNotIn((row, logical + q), seen)
                                    seen.add((row, logical + q))
                    self.assertEqual(
                        seen, {(r, c) for r in range(valid) for c in expected_cols}
                    )

    def test_batched_gather_keeps_every_payload_record_and_return_mapping(self):
        for hidden in (1280, 2048, 5120, 7936):
            words = hidden // 128
            record = (words + 2 + 3) // 4 * 4
            self.assertLessEqual(record, 64)
            for batch in (2, 4):
                for valid in (0, 1, 3, 4, 7, 31, 63, 127, 128):
                    rows = []
                    packets = []
                    metadata = []
                    returns = []
                    for base in range(0, 128, 4 * batch):
                        for wave in range(4):
                            for j in range(batch):
                                row = base + wave + j * 4
                                if row >= valid:
                                    continue
                                rows.append(row)
                                for lane in range(64):
                                    for packet in range((hidden + 1023) // 1024):
                                        byte = packet * 1024 + lane * 16
                                        if byte < hidden:
                                            packets.append((row, byte))
                                    if lane < record:
                                        metadata.append((row, lane))
                                returns.append(row)
                    self.assertEqual(sorted(rows), list(range(valid)))
                    self.assertEqual(
                        sorted(packets),
                        [(r, b) for r in range(valid) for b in range(0, hidden, 16)],
                    )
                    self.assertEqual(
                        sorted(metadata),
                        [(r, w) for r in range(valid) for w in range(record)],
                    )
                    self.assertEqual(sorted(returns), list(range(valid)))
            for ranks, experts in ((2, 3), (8, 48)):
                table = 3 * ranks * experts + ranks + 2 * experts + 2
                begin = (table + 7) // 8 * 8
                self.assertGreaterEqual(begin, table)
                self.assertEqual(begin % 8, 0)
                for row in range(128):
                    source, token, slot = (row % ranks, row * 7 % 1536, row % 6)
                    address = 140737219919872 + (slot * 1536 + token) * hidden * 2
                    low, high = (address & 4294967295, address >> 32)
                    self.assertEqual(low | high << 32, address)
                    code = (source * 1536 + token) * 6 + slot
                    self.assertEqual(divmod(code, 6), (source * 1536 + token, slot))

    def test_early_replacement_preserves_both_shared_and_routed_operand_rings(self):
        for mf, nf, stages, dmas in ((2, 4, 2, 14), (4, 4, 3, 15)):
            reads = mf + nf
            half = mf * nf
            next_read = 2 * half - reads
            self.assertGreaterEqual(next_read, half)
            for panels in (8, 18, 40):
                pending = [(k, k % stages) for k in range(stages)]
                ready = [None] * stages
                seen = []

                def wait(bound):
                    while len(pending) * dmas > bound:
                        logical, slot = pending.pop(0)
                        ready[slot] = logical

                wait((stages - 1) * dmas)
                for k in range(panels):
                    self.assertEqual(ready[k % stages], k)
                    seen.append(k)
                    issued_reads = list(range(half - reads, half))
                    self.assertEqual(len(issued_reads), reads)
                    self.assertLess(max(issued_reads), half)
                    if k + 1 < panels:
                        pending.append((k + stages, k % stages))
                        wait((stages - 1) * dmas)
                        self.assertEqual(ready[(k + 1) % stages], k + 1)
                wait(0)
                self.assertEqual(seen, list(range(panels)))

    def test_batched_route_reads_preserve_order_and_mask_every_partial_batch(self):
        rng = random.Random(321)
        for batch in (2, 4, 8):
            for entries in (0, 1, 255, 256, 257, 511, 512, 1023, 1024, 1025):
                routes = [rng.randrange(-1, 7) for _ in range(entries)]
                for expert in range(7):
                    selected = []
                    for first in range(0, entries, 256 * batch):
                        keys = [
                            [
                                routes[first + j * 256 + t]
                                if first + j * 256 + t < entries
                                else -1
                                for t in range(256)
                            ]
                            for j in range(batch)
                        ]
                        for j in range(batch):
                            selected.extend(
                                (
                                    first + j * 256 + t
                                    for t, key in enumerate(keys[j])
                                    if key == expert
                                )
                            )
                    self.assertEqual(
                        selected, [i for i, e in enumerate(routes) if e == expert]
                    )

    def test_gather_actors_rejoin_without_abandoning_inflight_rows_or_column_tickets(
        self,
    ):
        early_transfers = 0
        for reuse in (False, True):
            for actors in (8, 16):
                for producers in (1, 3):
                    for groups in (1, 2, 6):
                        if actors - producers < groups:
                            continue
                        for rows in (0, 1, 5, 27):
                            for seed in range(3):
                                rng = random.Random(seed + rows * 31 + groups)
                                pull = head = 0
                                ready = set()
                                gathered = []
                                finished = []
                                done = [set() for _ in range(rows)]
                                workers = {
                                    i: [
                                        "gather_claim" if i < producers else "claim",
                                        None,
                                    ]
                                    for i in range(actors)
                                }
                                while workers:
                                    eligible = []
                                    for i, (stage, item) in workers.items():
                                        if (
                                            stage == "wait"
                                            and item // groups not in ready
                                        ):
                                            continue
                                        if (
                                            stage == "join"
                                            and len(done[item // groups]) < groups
                                        ):
                                            continue
                                        eligible.append(i)
                                    self.assertTrue(
                                        eligible, "role transfer deadlocked"
                                    )
                                    i = rng.choice(eligible)
                                    stage, item = workers[i]
                                    if stage == "gather_claim":
                                        if pull < rows:
                                            workers[i] = ["gather_copy", pull]
                                            pull += 1
                                        elif reuse:
                                            early_transfers += len(ready) < rows
                                            workers[i] = ["claim", None]
                                        else:
                                            del workers[i]
                                    elif stage == "gather_copy":
                                        self.assertNotIn(item, ready)
                                        ready.add(item)
                                        gathered.append(item)
                                        workers[i] = ["gather_claim", None]
                                    elif stage == "claim":
                                        if head >= rows * groups:
                                            del workers[i]
                                        else:
                                            workers[i] = ["wait", head]
                                            head += 1
                                    elif stage == "wait":
                                        workers[i] = ["first", item]
                                    elif stage == "first":
                                        done[item // groups].add(item % groups)
                                        workers[i] = ["join", item]
                                    elif stage == "join":
                                        workers[i] = ["second", item]
                                    else:
                                        self.assertEqual(
                                            len(done[item // groups]), groups
                                        )
                                        finished.append(item)
                                        workers[i] = ["claim", None]
                                self.assertEqual(sorted(gathered), list(range(rows)))
                                self.assertEqual(
                                    sorted(finished), list(range(rows * groups))
                                )
        self.assertGreater(early_transfers, 0)

    def test_control_relay_releases_only_after_all_current_participants_arrive(self):
        for start in (0, 2**29 - 3):
            for seed in range(8):
                rng = random.Random(seed)
                flags = [0] * 257
                committed = 12345
                for frame, (actors, relay) in enumerate(
                    (
                        (129, False),
                        (33, True),
                        (257, True),
                        (17, False),
                        (65, True),
                        (129, False),
                        (33, True),
                    )
                ):
                    base = (start + frame) * 8 % 2**32
                    tags = [(base + p) % 2**32 for p in range(1, 6)]
                    if relay:
                        committed = tags[1]
                        committed = 0
                    phases = [0] * actors
                    waiting = [False] * actors
                    arrived = [set() for _ in tags]
                    released = [set() for _ in tags]
                    steps = 0
                    while min(phases) < len(tags):
                        steps += 1
                        self.assertLess(steps, actors * len(tags) * 100)
                        actor = rng.randrange(actors)
                        phase = phases[actor]
                        if phase == len(tags):
                            continue
                        tag = tags[phase]
                        if not waiting[actor]:
                            flags[actor] = tag
                            arrived[phase].add(actor)
                            waiting[actor] = True
                            continue
                        relayed = relay and phase in (1, 2, 4)
                        if relayed:
                            if actor == 0 and all((f == tag for f in flags[:actors])):
                                committed = tag
                            ready = committed == tag
                        else:
                            ready = all(
                                (
                                    f and (f - tag) % 2**32 < 2**31
                                    for f in flags[:actors]
                                )
                            )
                        if ready:
                            self.assertEqual(arrived[phase], set(range(actors)))
                            released[phase].add(actor)
                            phases[actor] += 1
                            waiting[actor] = False
                    self.assertTrue(all((len(r) == actors for r in released)))

    def test_coalesced_weight_packets_exchange_to_identical_original_fragments(self):
        packets = [(lane // 4, lane % 4) for lane in range(64)]
        self.assertEqual(set(packets), {(r, g) for r in range(16) for g in range(4)})
        for lane in range(64):
            r, g = (lane % 16, lane // 16)
            self.assertEqual(packets[r * 4 + g], (r, g))


if __name__ == "__main__":
    unittest.main()
