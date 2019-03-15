"""Code referenced from open source implementation at https://github.com/rtqichen/torchdiffeq/, code is taken as is without re-writing according to IFT6759 standards."""
import sys
import collections
from .solvers import FixedGridODESolver
from .misc import _scaled_dot_product, _has_converged
from . import rk_common

_BASHFORTH_COEFFICIENTS = [
    [],  # order 0
    [11],
    [3, -1],
    [23, -16, 5],
    [55, -59, 37, -9],
    [1901, -2774, 2616, -1274, 251],
    [4277, -7923, 9982, -7298, 2877, -475],
    [198721, -447288, 705549, -688256, 407139, -134472, 19087],
    [434241, -1152169, 2183877, -2664477, 2102243, -1041723, 295767, -36799],
    [
        14097247,
        -43125206,
        95476786,
        -139855262,
        137968480,
        -91172642,
        38833486,
        -9664106,
        1070017,
    ],
    [
        30277247,
        -104995189,
        265932680,
        -454661776,
        538363838,
        -444772162,
        252618224,
        -94307320,
        20884811,
        -2082753,
    ],
    [
        2132509567,
        -8271795124,
        23591063805,
        -46113029016,
        63716378958,
        -63176201472,
        44857168434,
        -22329634920,
        7417904451,
        -1479574348,
        134211265,
    ],
    [
        4527766399,
        -19433810163,
        61633227185,
        -135579356757,
        214139355366,
        -247741639374,
        211103573298,
        -131365867290,
        58189107627,
        -17410248271,
        3158642445,
        -262747265,
    ],
    [
        13064406523627,
        -61497552797274,
        214696591002612,
        -524924579905150,
        932884546055895,
        -1233589244941764,
        1226443086129408,
        -915883387152444,
        507140369728425,
        -202322913738370,
        55060974662412,
        -9160551085734,
        703604254357,
    ],
    [
        27511554976875,
        -140970750679621,
        537247052515662,
        -1445313351681906,
        2854429571790805,
        -4246767353305755,
        4825671323488452,
        -4204551925534524,
        2793869602879077,
        -1393306307155755,
        505586141196430,
        -126174972681906,
        19382853593787,
        -1382741929621,
    ],
    [
        173233498598849,
        -960122866404112,
        3966421670215481,
        -11643637530577472,
        25298910337081429,
        -41825269932507728,
        53471026659940509,
        -53246738660646912,
        41280216336284259,
        -24704503655607728,
        11205849753515179,
        -3728807256577472,
        859236476684231,
        -122594813904112,
        8164168737599,
    ],
    [
        362555126427073,
        -2161567671248849,
        9622096909515337,
        -30607373860520569,
        72558117072259733,
        -131963191940828581,
        187463140112902893,
        -210020588912321949,
        186087544263596643,
        -129930094104237331,
        70724351582843483,
        -29417910911251819,
        9038571752734087,
        -1934443196892599,
        257650275915823,
        -16088129229375,
    ],
    [
        192996103681340479,
        -1231887339593444974,
        5878428128276811750,
        -20141834622844109630,
        51733880057282977010,
        -102651404730855807942,
        160414858999474733422,
        -199694296833704562550,
        199061418623907202560,
        -158848144481581407370,
        100878076849144434322,
        -50353311405771659322,
        19338911944324897550,
        -5518639984393844930,
        1102560345141059610,
        -137692773163513234,
        8092989203533249,
    ],
    [
        401972381695456831,
        -2735437642844079789,
        13930159965811142228,
        -51150187791975812900,
        141500575026572531760,
        -304188128232928718008,
        518600355541383671092,
        -710171024091234303204,
        786600875277595877750,
        -706174326992944287370,
        512538584122114046748,
        -298477260353977522892,
        137563142659866897224,
        -49070094880794267600,
        13071639236569712860,
        -2448689255584545196,
        287848942064256339,
        -15980174332775873,
    ],
    [
        333374427829017307697,
        -2409687649238345289684,
        13044139139831833251471,
        -51099831122607588046344,
        151474888613495715415020,
        -350702929608291455167896,
        647758157491921902292692,
        -967713746544629658690408,
        1179078743786280451953222,
        -1176161829956768365219840,
        960377035444205950813626,
        -639182123082298748001432,
        343690461612471516746028,
        -147118738993288163742312,
        48988597853073465932820,
        -12236035290567356418552,
        2157574942881818312049,
        -239560589366324764716,
        12600467236042756559,
    ],
    [
        691668239157222107697,
        -5292843584961252933125,
        30349492858024727686755,
        -126346544855927856134295,
        399537307669842150996468,
        -991168450545135070835076,
        1971629028083798845750380,
        -3191065388846318679544380,
        4241614331208149947151790,
        -4654326468801478894406214,
        4222756879776354065593786,
        -3161821089800186539248210,
        1943018818982002395655620,
        -970350191086531368649620,
        387739787034699092364924,
        -121059601023985433003532,
        28462032496476316665705,
        -4740335757093710713245,
        498669220956647866875,
        -24919383499187492303,
    ],
]

_MOULTON_COEFFICIENTS = [
    [],  # order 0
    [1],
    [1, 1],
    [5, 8, -1],
    [9, 19, -5, 1],
    [251, 646, -264, 106, -19],
    [475, 1427, -798, 482, -173, 27],
    [19087, 65112, -46461, 37504, -20211, 6312, -863],
    [36799, 139849, -121797, 123133, -88547, 41499, -11351, 1375],
    [1070017, 4467094, -4604594, 5595358, -5033120, 3146338, -1291214, 312874, -33953],
    [
        2082753,
        9449717,
        -11271304,
        16002320,
        -17283646,
        13510082,
        -7394032,
        2687864,
        -583435,
        57281,
    ],
    [
        134211265,
        656185652,
        -890175549,
        1446205080,
        -1823311566,
        1710774528,
        -1170597042,
        567450984,
        -184776195,
        36284876,
        -3250433,
    ],
    [
        262747265,
        1374799219,
        -2092490673,
        3828828885,
        -5519460582,
        6043521486,
        -4963166514,
        3007739418,
        -1305971115,
        384709327,
        -68928781,
        5675265,
    ],
    [
        703604254357,
        3917551216986,
        -6616420957428,
        13465774256510,
        -21847538039895,
        27345870698436,
        -26204344465152,
        19058185652796,
        -10344711794985,
        4063327863170,
        -1092096992268,
        179842822566,
        -13695779093,
    ],
    [
        1382741929621,
        8153167962181,
        -15141235084110,
        33928990133618,
        -61188680131285,
        86180228689563,
        -94393338653892,
        80101021029180,
        -52177910882661,
        25620259777835,
        -9181635605134,
        2268078814386,
        -345457086395,
        24466579093,
    ],
    [
        8164168737599,
        50770967534864,
        -102885148956217,
        251724894607936,
        -499547203754837,
        781911618071632,
        -963605400824733,
        934600833490944,
        -710312834197347,
        418551804601264,
        -187504936597931,
        61759426692544,
        -14110480969927,
        1998759236336,
        -132282840127,
    ],
    [
        16088129229375,
        105145058757073,
        -230992163723849,
        612744541065337,
        -1326978663058069,
        2285168598349733,
        -3129453071993581,
        3414941728852893,
        -2966365730265699,
        2039345879546643,
        -1096355235402331,
        451403108933483,
        -137515713789319,
        29219384284087,
        -3867689367599,
        240208245823,
    ],
    [
        8092989203533249,
        55415287221275246,
        -131240807912923110,
        375195469874202430,
        -880520318434977010,
        1654462865819232198,
        -2492570347928318318,
        3022404969160106870,
        -2953729295811279360,
        2320851086013919370,
        -1455690451266780818,
        719242466216944698,
        -273894214307914510,
        77597639915764930,
        -15407325991235610,
        1913813460537746,
        -111956703448001,
    ],
    [
        15980174332775873,
        114329243705491117,
        -290470969929371220,
        890337710266029860,
        -2250854333681641520,
        4582441343348851896,
        -7532171919277411636,
        10047287575124288740,
        -10910555637627652470,
        9644799218032932490,
        -6913858539337636636,
        3985516155854664396,
        -1821304040326216520,
        645008976643217360,
        -170761422500096220,
        31816981024600492,
        -3722582669836627,
        205804074290625,
    ],
    [
        12600467236042756559,
        93965550344204933076,
        -255007751875033918095,
        834286388106402145800,
        -2260420115705863623660,
        4956655592790542146968,
        -8827052559979384209108,
        12845814402199484797800,
        -15345231910046032448070,
        15072781455122686545920,
        -12155867625610599812538,
        8008520809622324571288,
        -4269779992576330506540,
        1814584564159445787240,
        -600505972582990474260,
        149186846171741510136,
        -26182538841925312881,
        2895045518506940460,
        -151711881512390095,
    ],
    [
        24919383499187492303,
        193280569173472261637,
        -558160720115629395555,
        1941395668950986461335,
        -5612131802364455926260,
        13187185898439270330756,
        -25293146116627869170796,
        39878419226784442421820,
        -51970649453670274135470,
        56154678684618739939910,
        -50320851025594566473146,
        37297227252822858381906,
        -22726350407538133839300,
        11268210124987992327060,
        -4474886658024166985340,
        1389665263296211699212,
        -325187970422032795497,
        53935307402575440285,
        -5652892248087175675,
        281550972898020815,
    ],
]

_DIVISOR = [
    None,
    11,
    2,
    12,
    24,
    720,
    1440,
    60480,
    120960,
    3628800,
    7257600,
    479001600,
    958003200,
    2615348736000,
    5230697472000,
    31384184832000,
    62768369664000,
    32011868528640000,
    64023737057280000,
    51090942171709440000,
    102181884343418880000,
]

_MIN_ORDER = 4
_MAX_ORDER = 12
_MAX_ITERS = 4


class AdamsBashforthMoulton(FixedGridODESolver):
    def __init__(
        self,
        func,
        y0,
        rtol=1e-3,
        atol=1e-4,
        implicit=True,
        max_iters=_MAX_ITERS,
        max_order=_MAX_ORDER,
        **kwargs
    ):
        super(AdamsBashforthMoulton, self).__init__(func, y0, **kwargs)

        self.rtol = rtol
        self.atol = atol
        self.implicit = implicit
        self.max_iters = max_iters
        self.max_order = int(min(max_order, _MAX_ORDER))
        self.prev_f = collections.deque(maxlen=self.max_order - 1)
        self.prev_t = None

    def _update_history(self, t, f):
        if self.prev_t is None or self.prev_t != t:
            self.prev_f.appendleft(f)
            self.prev_t = t

    def step_func(self, func, t, dt, y):
        self._update_history(t, func(t, y))
        order = min(len(self.prev_f), self.max_order - 1)
        if order < _MIN_ORDER - 1:
            # Compute using RK4.
            dy = rk_common.rk4_alt_step_func(func, t, dt, y, k1=self.prev_f[0])
            return dy
        else:
            # Adams-Bashforth predictor.
            bashforth_coeffs = _BASHFORTH_COEFFICIENTS[order]
            ab_div = _DIVISOR[order]
            dy = tuple(
                dt * _scaled_dot_product(1 / ab_div, bashforth_coeffs, f_)
                for f_ in zip(*self.prev_f)
            )

            # Adams-Moulton corrector.
            if self.implicit:
                moulton_coeffs = _MOULTON_COEFFICIENTS[order + 1]
                am_div = _DIVISOR[order + 1]
                delta = tuple(
                    dt * _scaled_dot_product(1 / am_div, moulton_coeffs[1:], f_)
                    for f_ in zip(*self.prev_f)
                )
                converged = False
                for _ in range(self.max_iters):
                    dy_old = dy
                    f = func(t + dt, tuple(y_ + dy_ for y_, dy_ in zip(y, dy)))
                    dy = tuple(
                        dt * (moulton_coeffs[0] / am_div) * f_ + delta_
                        for f_, delta_ in zip(f, delta)
                    )
                    converged = _has_converged(dy_old, dy, self.rtol, self.atol)
                    if converged:
                        break
                if not converged:
                    print(
                        "Warning: Functional iteration did not converge. Solution may be incorrect.",
                        file=sys.stderr,
                    )
                    self.prev_f.pop()
                self._update_history(t, f)
            return dy

    @property
    def order(self):
        return 4


class AdamsBashforth(AdamsBashforthMoulton):
    def __init__(self, func, y0, **kwargs):
        super(AdamsBashforth, self).__init__(func, y0, implicit=False, **kwargs)
