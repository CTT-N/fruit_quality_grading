"""
MODULE 1 - THEO DÕI QUẢ QUA CÁC FRAME (centroid tracking)
Mỗi quả có 1 track_id cố định trong suốt thời gian đi qua băng chuyền,
để M3 chỉ phân loại 1 lần/quả và bỏ phiếu đa số qua nhiều frame.
"""
import numpy as np
import config


class CentroidTracker:
    def __init__(self, max_dist=None, max_missing=None):
        self.max_dist = max_dist or config.TRACKER["max_dist"]
        self.max_missing = max_missing if max_missing is not None else config.TRACKER["max_missing"]
        self.tracks = {}       # id -> dict(centroid, missing)
        self.next_id = 0

    def update(self, objects):
        """
        Input : List[FruitObject] của frame hiện tại (track_id = -1)
        Output: chính list đó, đã gán track_id (khớp theo khoảng cách tâm gần nhất, tham lam)
        """
        ids = list(self.tracks.keys())
        pairs = []
        for j, o in enumerate(objects):
            for tid in ids:
                d = float(np.hypot(o.centroid[0] - self.tracks[tid]["centroid"][0],
                                   o.centroid[1] - self.tracks[tid]["centroid"][1]))
                if d <= self.max_dist:
                    pairs.append((d, j, tid))
        pairs.sort()
        used_obj, used_tid = set(), set()
        for d, j, tid in pairs:
            if j in used_obj or tid in used_tid:
                continue
            objects[j].track_id = tid
            self.tracks[tid] = dict(centroid=objects[j].centroid, missing=0)
            used_obj.add(j); used_tid.add(tid)

        for j, o in enumerate(objects):            # quả mới
            if j not in used_obj:
                o.track_id = self.next_id
                self.tracks[self.next_id] = dict(centroid=o.centroid, missing=0)
                self.next_id += 1

        for tid in ids:                            # track mất dấu
            if tid not in used_tid:
                self.tracks[tid]["missing"] += 1
                if self.tracks[tid]["missing"] > self.max_missing:
                    del self.tracks[tid]
        return objects
