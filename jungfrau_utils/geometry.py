from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DetectorGeometry:
    origin_y: tuple[int, ...] = (0,)
    origin_x: tuple[int, ...] = (0,)
    mod_rot90: tuple[int, ...] = (0,)
    is_stripsel: bool = False
    det_rot90: int = 0

    def __post_init__(self):
        if self.mod_rot90 == (0,) and len(self.origin_y) > 1:
            self.mod_rot90 *= len(self.origin_y)


# fmt: off
detector_geometry = {
    "JF01T03V01": DetectorGeometry(
        origin_y=(0, 550, 1100),
        origin_x=(0, 0, 0),
    ),

    "JF02T09V01": DetectorGeometry(
        origin_y=(0, 0, 0, 0, 0, 0, 0, 0, 0),
        origin_x=(0, 1036, 2072, 3108, 4144, 5180, 6216, 7252, 8288),
    ),

    "JF02T09V02": DetectorGeometry(
        origin_y=(0, 0, 0, 0, 0, 0, 0, 0, 0),
        origin_x=(8288, 7252, 6216, 5180, 4144, 3108, 2072, 1036, 0),
        det_rot90=2,
    ),

    "JF02T09V03": DetectorGeometry(
        origin_y=(0, 0, 0, 0, 0, 0, 0, 0, 0),
        origin_x=(8288, 7252, 6216, 5180, 4144, 3108, 2072, 1036, 0),
        det_rot90=2,
    ),

    "JF02T01V02": DetectorGeometry(det_rot90=2),

    "JF03T01V01": DetectorGeometry(),

    "JF03T01V02": DetectorGeometry(),

    "JF04T01V01": DetectorGeometry(),

    "JF05T01V01": DetectorGeometry(
        is_stripsel=True
    ),

    "JF06T32V01": DetectorGeometry(
        origin_y=(
            68, 0, 618, 618,
            550, 550, 1168, 1168,
            1100, 1100, 1718, 1718,
            1650, 1650, 2268, 2268,
            2200, 2200, 2818, 2818,
            2750, 2750, 3368, 3368,
            3300, 3300, 3918, 3918,
            3850, 3850, 4468, 4400,
        ),
        origin_x=(
            972, 2011, 0, 1039,
            2078, 3117, 0, 1039,
            2078, 3117, 0, 1039,
            2078, 3117, 66, 1106,
            2145, 3184, 66, 1106,
            2145, 3184, 66, 1106,
            2145, 3184, 66, 1106,
            2145, 3184, 1106, 2145,
        ),
        det_rot90=1,
    ),

    "JF06T32V02": DetectorGeometry(
        origin_y=(
            68, 0, 618, 618,
            550, 550, 1168, 1168,
            1100, 1100, 1718, 1718,
            1650, 1650, 2268, 2268,
            2200, 2200, 2818, 2818,
            2750, 2750, 3368, 3368,
            3300, 3300, 3918, 3918,
            3850, 3850, 4468, 4400,
        ),
        origin_x=(
            972, 2011, 0, 1039,
            2078, 3117, 0, 1039,
            2078, 3117, 0, 1039,
            2078, 3117, 66, 1106,
            2145, 3184, 66, 1106,
            2145, 3184, 66, 1106,
            2145, 3184, 66, 1106,
            2145, 3184, 1106, 2145,
        ),
        det_rot90=1,
    ),

    "JF06T32V03": DetectorGeometry(
        origin_y=(
            68, 0, 618, 618,
            550, 550, 1168, 1168,
            1100, 1100, 1718, 1718,
            1650, 1650, 2268, 2268,
            2200, 2200, 2818, 2818,
            2750, 2750, 3368, 3368,
            3300, 3300, 3918, 3918,
            3850, 3850, 4468, 4400,
        ),
        origin_x=(
            972, 2011, 0, 1039,
            2078, 3117, 0, 1039,
            2078, 3117, 0, 1039,
            2078, 3117, 66, 1106,
            2145, 3184, 66, 1106,
            2145, 3184, 66, 1106,
            2145, 3184, 66, 1106,
            2145, 3184, 1106, 2145,
        ),
        det_rot90=1,
    ),

    "JF06T32V04": DetectorGeometry(
        origin_y=(
            68, 0, 618, 618,
            550, 550, 1168, 1168,
            1100, 1100, 1718, 1718,
            1650, 1650, 2268, 2268,
            2200, 2200, 2818, 2818,
            2750, 2750, 3368, 3368,
            3300, 3300, 3918, 3918,
            3850, 3850, 4468, 4400,
        ),
        origin_x=(
            972, 2011, 0, 1039,
            2078, 3117, 0, 1039,
            2078, 3117, 0, 1039,
            2078, 3117, 66, 1106,
            2145, 3184, 66, 1106,
            2145, 3184, 66, 1106,
            2145, 3184, 66, 1106,
            2145, 3184, 1106, 2145,
        ),
        det_rot90=1,
    ),

    "JF06T08V01": DetectorGeometry(
        origin_y=(
            68, 0, 618, 550,
            1168, 1100, 1718, 1650,
        ),
        origin_x=(
            0, 1039, 0, 1039,
            67, 1106, 67, 1106,
        ),
        det_rot90=1,
    ),

    "JF06T08V02": DetectorGeometry(
        origin_y=(
            68, 0, 618, 550,
            1168, 1100, 1718, 1650,
        ),
        origin_x=(
            0, 1039, 0, 1039,
            67, 1106, 67, 1106,
        ),
        det_rot90=1,
    ),

    "JF06T08V03": DetectorGeometry(
        origin_y=(
            68, 0, 618, 550,
            1168, 1100, 1718, 1650,
        ),
        origin_x=(
            0, 1039, 0, 1039,
            67, 1106, 67, 1106,
        ),
        det_rot90=1,
    ),

    "JF06T08V04": DetectorGeometry(
        origin_y=(
            68, 0, 618, 550,
            1168, 1100, 1718, 1650,
        ),
        origin_x=(
            0, 1039, 0, 1039,
            67, 1106, 67, 1106,
        ),
        det_rot90=1,
    ),

    "JF06T08V05": DetectorGeometry(
        origin_y=(
            68, 0, 618, 550,
            1168, 1100, 1718, 1650,
        ),
        origin_x=(
            0, 1039, 0, 1039,
            67, 1106, 67, 1106,
        ),
        det_rot90=1,
    ),

    "JF06T04V01": DetectorGeometry(
        origin_y=(68, 0, 618, 550),
        origin_x=(0, 1039, 67, 1106),
        det_rot90=1,
    ),

    "JF07T32V01": DetectorGeometry(
        origin_y=(
            0, 0, 68, 68,
            550, 550, 618, 618,
            1100, 1100, 1168, 1168,
            1650, 1650, 1718, 1718,
            2200, 2200, 2268, 2268,
            2750, 2750, 2818, 2818,
            3300, 3300, 3368, 3368,
            3850, 3850, 3918, 3918,
        ),
        origin_x=(
            68, 1107, 2146, 3185,
            68, 1107, 2146, 3185,
            68, 1107, 2146, 3185,
            68, 1107, 2146, 3185,
            0, 1039, 2078, 3117,
            0, 1039, 2078, 3117,
            0, 1039, 2078, 3117,
            0, 1039, 2078, 3117,
        ),
    ),

    "JF07T32V02": DetectorGeometry(
        origin_y=(
            0, 0, 68, 68,
            550, 550, 618, 618,
            1100, 1100, 1168, 1168,
            1650, 1650, 1718, 1718,
            2200, 2200, 2268, 2268,
            2750, 2750, 2818, 2818,
            3300, 3300, 3368, 3368,
            3850, 3850, 3918, 3918,
        ),
        origin_x=(
            68, 1107, 2146, 3185,
            68, 1107, 2146, 3185,
            68, 1107, 2146, 3185,
            68, 1107, 2146, 3185,
            0, 1039, 2078, 3117,
            0, 1039, 2078, 3117,
            0, 1039, 2078, 3117,
            0, 1039, 2078, 3117,
        ),
    ),

    "JF07T03V01": DetectorGeometry(
        origin_y=(0, 550, 1100),
        origin_x=(0, 0, 0),
    ),

    "JF08T01V01": DetectorGeometry(),

    "JF09T01V01": DetectorGeometry(),

    "JF10T01V01": DetectorGeometry(
        is_stripsel=True,
    ),

    "JF11T04V01": DetectorGeometry(
        origin_y=(0, 0, 0, 0),
        origin_x=(9306, 6198, 3108, 0),
        is_stripsel=True,
    ),

    "JF12T04V01": DetectorGeometry(
        # a vertical gap of 4 pix is artificial, because they are physically separated 2 detectors
        origin_y=(0, 0, 90, 90),
        origin_x=(0, 3117, 0, 3117),
        is_stripsel=True,
    ),

    "JF13T01V01": DetectorGeometry(),

    "JF14T01V01": DetectorGeometry(),

    "JF15T08V01": DetectorGeometry(
        origin_y=(0, 149, 550, 699, 1100, 1249, 1649, 1798),
        origin_x=(149, 1185, 149, 1185, 0, 1037, 0, 1037),
    ),

    "JF16T03V01": DetectorGeometry(
        origin_y=(0, 550, 1100),
        origin_x=(0, 0, 0),
    ),

    "JF16T03V02": DetectorGeometry(
        origin_y=(0, 550, 1100),
        origin_x=(0, 0, 0),
    ),

    "JF17T16V01": DetectorGeometry(
        origin_y=(
            0, 69, 550, 619,
            1100, 1169, 1650, 1719,
            2200, 2269, 2750, 2819,
            597, 667, 1636, 1706,
        ),
        origin_x=(
            607, 1646, 607, 1646,
            607, 1646, 538, 1577,
            538, 1577, 538, 1577,
            583, 3212, 514, 3143,
        ),
        mod_rot90=(
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            3, 3, 3, 3,
        ),
    ),

    "JF18T01V01": DetectorGeometry(
        is_stripsel=True,
    ),
}
# fmt: on
