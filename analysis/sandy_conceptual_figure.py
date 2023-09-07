import copy
import gzip
import math
import pickle
from functools import partial
import matplotlib as mpl
import matplotlib.pyplot as plt
import shapefile
from matplotlib.patches import ConnectionPatch, Circle, PathPatch
from matplotlib.colors import Normalize
from matplotlib.transforms import Affine2D
from scipy.interpolate import interp1d
from shapely.geometry import Point
import pyproj
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from shapely.ops import transform
import pandas as pd
import numpy as np

MAX_COLUMN_HEIGHT = 9.60
MAX_FIG_WIDTH_WIDE = 7.07
MAX_FIG_WIDTH_NARROW = 3.45
FSIZE_TINY = 6
FSIZE_SMALL = 8
FSIZE_MEDIUM = 10
FSIZE_LARGE = 12

plt.rc('font', size=FSIZE_SMALL)  # controls default text sizes
plt.rc('axes', titlesize=FSIZE_SMALL)  # fontsize of the axes title
plt.rc('axes', labelsize=FSIZE_SMALL)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=FSIZE_SMALL)  # fontsize of the tick labels
plt.rc('ytick', labelsize=FSIZE_SMALL)  # fontsize of the tick labels
plt.rc('legend', fontsize=FSIZE_SMALL)  # legend fontsize
plt.rc('figure', titlesize=FSIZE_LARGE)  # fontsize of the figure title
plt.rc('axes', linewidth=0.5)  # fontsize of the figure title

billion_dollar_damage_ranges = {
    'US.NY': {
        'min': 20,
        'max': 50,
    },
    'US.NJ': {
        'min': 20,
        'max': 50,
    },
    'US.CT': {
        'min': 1,
        'max': 2,
    },
    'US.PA': {
        'min': 1,
        'max': 2,
    },
    'US.VA': {
        'min': 0.25,
        'max': 0.5,
    },
    'US.MA': {
        'min': 0.25,
        'max': 0.5,
    },
    'US.MD': {
        'min': 0.25,
        'max': 0.5,
    },
    'US.OH': {
        'min': 0.25,
        'max': 0.5,
    },
    'US.RI': {
        'min': 0.25,
        'max': 0.5,
    },
    'US.DE': {
        'min': 0.05,
        'max': 0.25,
    },
    'US.NH': {
        'min': 0.05,
        'max': 0.25,
    },
    'US.WV': {
        'min': 0.05,
        'max': 0.25,
    },
    'US.NC': {
        'min': 0.05,
        'max': 0.25,
    }
}
billion_dollar_damage_total_max = 107.5
billion_dollar_damage_total_min = 43.45

prop_cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
us_state_colors = {
    'US.NJ': prop_cycle_colors[0],
    'US.NY': prop_cycle_colors[1],
}

world_regions = {
    'USA': [
        'US.AL',
        'US.AK',
        'US.AZ',
        'US.AR',
        'US.CA',
        'US.CO',
        'US.CT',
        'US.DE',
        'US.DC',
        'US.FL',
        'US.GA',
        'US.HI',
        'US.ID',
        'US.IL',
        'US.IN',
        'US.IA',
        'US.KS',
        'US.KY',
        'US.LA',
        'US.ME',
        'US.MD',
        'US.MA',
        'US.MI',
        'US.MN',
        'US.MS',
        'US.MO',
        'US.MT',
        'US.NE',
        'US.NV',
        'US.NH',
        'US.NJ',
        'US.NM',
        'US.NY',
        'US.NC',
        'US.ND',
        'US.OH',
        'US.OK',
        'US.OR',
        'US.PA',
        'US.RI',
        'US.SC',
        'US.SD',
        'US.TN',
        'US.TX',
        'US.UT',
        'US.VT',
        'US.VA',
        'US.WA',
        'US.WV',
        'US.WI',
        'US.WY'
    ],
    'CHN': [
        'CN.AH',
        'CN.BJ',
        'CN.CQ',
        'CN.FJ',
        'CN.GS',
        'CN.GD',
        'CN.GX',
        'CN.GZ',
        'CN.HA',
        'CN.HB',
        'CN.HL',
        'CN.HE',
        'CN.HU',
        'CN.HN',
        'CN.JS',
        'CN.JX',
        'CN.JL',
        'CN.LN',
        'CN.NM',
        'CN.NX',
        'CN.QH',
        'CN.SA',
        'CN.SD',
        'CN.SH',
        'CN.SX',
        'CN.SC',
        'CN.TJ',
        'CN.XJ',
        'CN.XZ',
        'CN.YN',
        'CN.ZJ'
    ],
    'MEX': ['MEX'],
    'CAN': ['CAN'],
    'EUR': [
        'BGR',
        'FIN',
        'ROU',
        'BEL',
        'GBR',
        'HUN',
        'BLR',
        'GRC',
        'AND',
        'ANT',
        'NOR',
        'SMR',
        'MDA',
        'SRB',
        'LTU',
        'SWE',
        'AUT',
        'ALB',
        'MKD',
        'UKR',
        'CHE',
        'LIE',
        'PRT',
        'SVN',
        'SVK',
        'HRV',
        'DEU',
        'NLD',
        'MNE',
        'LVA',
        'IRL',
        'CZE',
        'LUX',
        'ISL',
        'FRA',
        'DNK',
        'ITA',
        'CYP',
        'BIH',
        'POL',
        'EST',
        'ESP',
        'MLT',
        'MCO'
    ],
}

can_mex_patchespicklefile = "../data/external/maps/patchespickle_can_mex.pkl.gz"
us_state_patchespicklefile = "../data/external/maps/usa_robin.pkl.gz"
affected_counties_patchespicklefile = "../data/external/maps/patchespickle_affected_counties_SANDY__robin.pkl.gz"
world_patchespicklefile = "../data/external/maps/map_robinson_0.1simplified.pkl.gz"

world_patchespickle = pickle.load(gzip.GzipFile(world_patchespicklefile, "rb"))
world_projectionstr = world_patchespickle["projection"]


def my_transform(scale, t, trans, x, y):
    p = trans(x, y)
    return (p[0] * scale + t[0], p[1] * scale + t[1])


def get_projection(projectionstr):
    return partial(
        pyproj.transform,
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
        pyproj.Proj("+proj=%s" % projectionstr),
    )  # e.g. proj=eqc, proj=cea, http://geotiff.maptools.org/proj_list/


def fill_axis(_ax, _inset=True):
    _ax.add_collection(PatchCollection(
        world_region_patches,
        edgecolors="white",
        facecolors=world_region_colors,
        # linewidths=0.2,
        linewidths=world_region_linewidths,
        rasterized=True,
    ))
    # ax.add_collection(PatchCollection(usa_rest_patches, facecolor='lightgray'))
    # ax.add_collection(PatchCollection(can_mex_patches, facecolor='lightgray'))
    _ax.add_collection(PatchCollection(usa_affected_patches, edgecolors='white', linewidths=0.15,
                                       facecolors='gray'))
    _ax.add_collection(PatchCollection(usa_nj_ny_patches, edgecolors='k', linewidths=0.3,
                                       facecolors=usa_nj_ny_fc))
    if _inset:
        _ax.add_collection(
            PatchCollection(affected_counties_patches, hatch='//////////', facecolors='None', edgecolors='k',
                            linewidths=0.1))
    # _ax.add_collection(
    #     PatchCollection(usa_affected_patches, facecolor='None', edgecolors=usa_affected_patches_ec, linewidths=usa_affected_patches_lw))
    _ax.plot(sandy_x, sandy_y, linestyle='dotted', color='k')


translation_projection = get_projection('robin')
world_projection = get_projection(world_projectionstr)
projection = get_projection(
    "aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
)

eur_offset_x, eur_offset_y = -1.4e6, 1.0e6
chn_offset_x, chn_offset_y = -9.5e6, -1.9e6
can_offset_x, can_offset_y = 0, 2e4
mex_offset_x, mex_offset_y = 0, -2e4
world_region_patches = []
world_region_colors = []
world_region_linewidths = []
for n, (_, _, p) in zip(world_patchespickle["patches"].keys(), world_patchespickle['patches'].values()):
    print('{}'.format(n))
    for r in world_regions.keys():
        if n in world_regions[r]:
            if r == "EUR":
                p.set_transform(Affine2D.from_values(1, 0, 0, 1, eur_offset_x, eur_offset_y))
            elif r == "CHN":
                p.set_transform(Affine2D.from_values(1, 0, 0, 1, chn_offset_x, chn_offset_y))
            # elif r == "MEX":
            #     p.set_transform(Affine2D.from_values(1, 0, 0, 1, mex_offset_x, mex_offset_y))
            # elif r == "CAN":
            #     p.set_transform(Affine2D.from_values(1, 0, 0, 1, can_offset_x, can_offset_y))
            world_region_patches.append(p)
            world_region_colors.append('lightgray')
            if n not in ['CAN', 'MEX']:
                world_region_linewidths.append(0.15)
            else:
                world_region_linewidths.append(1)

us_projection_offset_lon = -96
us_projection_offset_lat = 37.5

affected_states = list(billion_dollar_damage_ranges.keys())
# affected_states = ['US.NJ', 'US.NY']

can_mex_patches = [rec[2] for rec in pickle.load(gzip.GzipFile(can_mex_patchespicklefile, "rb"))['patches'].values()]

usa_affected_patches = []
usa_rest_patches = []
usa_nj_ny_patches = []
usa_nj_ny_fc = []
for rec in pickle.load(gzip.GzipFile(us_state_patchespicklefile, "rb"))['patches'].values():
    state_name = list(rec[1])[0]
    patch = rec[2]
    if state_name == 'US.AK':
        patch.set_transform(patch.get_transform() + mpl.transforms.Affine2D().scale(
            0.4) + mpl.transforms.ScaledTranslation(transform(translation_projection, Point(-25, 0)).x,
                                                    transform(translation_projection, Point(0, -14)).y,
                                                    patch.get_transform()))  #
    elif state_name == 'US.HI':
        patch.set_transform(
            patch.get_transform() + mpl.transforms.ScaledTranslation(
                transform(translation_projection, Point(-15, 0)).x,
                transform(translation_projection, Point(0, -18)).y,
                patch.get_transform()))
    if state_name in affected_states:
        if state_name in us_state_colors:
            usa_nj_ny_fc.append(us_state_colors[state_name])
            usa_nj_ny_patches.append(patch)
        else:
            # usa_affected_patches_fc.append('gray')
            # usa_affected_patches_ec.append('white')
            # usa_affected_patches_lw.append(0.15)
            usa_affected_patches.append(patch)
    else:
        usa_rest_patches.append(patch)

affected_counties_patches = [rec[2] for rec in
                             pickle.load(gzip.GzipFile(affected_counties_patchespicklefile, "rb"))['patches'].values()]

sf = shapefile.Reader("../data/external/maps/IBTrACS.NA.list.v04r00.points/IBTrACS.NA.list.v04r00.points.shp")
sandy_shape_trafo = []
sandy_shape = []
for shaperec in sf.iterShapeRecords():
    rec = shaperec.record
    if rec[0] == '2012296N14283':
        coords = shaperec.shape.__geo_interface__['coordinates']
        # point = Point(coords[0] - us_projection_offset_lon, coords[1] - us_projection_offset_lat)
        point = Point(coords[0], coords[1])
        sandy_shape.append(point)
        sandy_shape_trafo.append(transform(world_projection, point))
sandy_x = [point.x for point in sandy_shape_trafo]
sandy_y = [point.y for point in sandy_shape_trafo]
sandy_track = interp1d(sandy_x, sandy_y)

fig, ax = plt.subplots(figsize=(MAX_FIG_WIDTH_WIDE, MAX_FIG_WIDTH_WIDE * 0.56))
fill_axis(ax, _inset=False)
xmin, xmax, ymin, ymax = -1.3e7, 2e6, 0.0e6, 8.4e6
# ax.set_yticks([])
# ax.set_xticks([])
ax.axis('off')
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

inset_scale = 4
axins = zoomed_inset_axes(ax, inset_scale, loc='upper left', borderpad=0)
fill_axis(axins)
axins.set_facecolor('w')
inset_x1, inset_x2, inset_y1, inset_y2 = -6.9e6, -6e6, 3.9e6, 4.85e6
axins.set_xlim(inset_x1, inset_x2)
axins.set_ylim(inset_y1, inset_y2)
axins.set_xticks([])
axins.set_yticks([])
# axins.axis('off')
# ax.indicate_inset_zoom(axins)
ax.add_patch(mpl.patches.Rectangle(
    (inset_x1, inset_y1),
    inset_x2 - inset_x1,
    inset_y2 - inset_y1,
    facecolor='None',
    edgecolor='k',
    linewidth=0.5))
ax.add_artist(ConnectionPatch((0, 0), (inset_x1, inset_y1), 'axes fraction', 'data', axins, ax, linewidth=0.5))
ax.add_artist(ConnectionPatch((1, 1), (inset_x2, inset_y2), 'axes fraction', 'data', axins, ax, linewidth=0.5))


class PPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def norm(self):
        l = self.length()
        return PPoint(self.x / l, self.y / l)

    def __getitem__(self, i):
        return self.x if i == 0 else self.y

    def __repr__(self):
        return "({}, {})".format(self.x, self.y)

    def __mul__(self, a):
        return PPoint(self.x * a, self.y * a)

    def __truediv__(self, a):
        return PPoint(self.x / a, self.y / a)

    def __add__(self, a):
        return PPoint(self.x + a.x, self.y + a.y)

    def __sub__(self, a):
        return PPoint(self.x - a.x, self.y - a.y)

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def __mod__(self, a):
        return PPoint(a.y - self.y, self.x - a.x).norm()


class PPath:
    def __init__(self, points=None, operations=None):
        self.points = [PPoint(p[0], p[1]) for p in points] if points is not None else []
        self.operations = [o for o in operations] if operations is not None else []
        if len(self.points) > len(self.operations):
            self.operations += [Path.MOVETO] * (len(self.points) - len(self.operations))

    def __add__(self, a):
        if isinstance(a, PPath):
            return PPath(self.points + a.points, self.operations + a.operations)
        elif isinstance(a, Path):
            return PPath(
                self.points + [PPoint(p[0], p[1]) for p in a.vertices],
                self.operations + list(a.codes),
            )
        elif isinstance(a, PPoint):
            return PPath([p + a for p in self.points], self.operations)

    def __mul__(self, a):
        return PPath([p * a for p in self.points], self.operations)

    def from_matplotlib(a):
        return PPath(a.vertices, a.codes)

    def to_matplotlib(self):
        return Path([[p.x, p.y] for p in self.points], self.operations)

    def __repr__(self):
        return ", ".join(
            ["{} {}".format(o, p) for p, o in zip(self.points, self.operations)]
        )

    def __getitem__(self, i):
        return self.points[i]


# form https://stackoverflow.com/questions/29465468/python-intersection-point-of-two-great-circles-lat-long
def intersect(circle1, circle2):
    p1_lat1_rad = (math.pi * circle1[0].y) / 180.0
    p1_lon1_rad = (math.pi * circle1[0].x) / 180.0
    p1_lat2_rad = (math.pi * circle1[1].y) / 180.0
    p1_lon2_rad = (math.pi * circle1[1].x) / 180.0

    p2_lat1_rad = (math.pi * circle2[0].y) / 180.0
    p2_lon1_rad = (math.pi * circle2[0].x) / 180.0
    p2_lat2_rad = (math.pi * circle2[1].y) / 180.0
    p2_lon2_rad = (math.pi * circle2[1].x) / 180.0

    # polar coordinates
    x1 = math.cos(p1_lat1_rad) * math.cos(p1_lon1_rad)
    y1 = math.cos(p1_lat1_rad) * math.sin(p1_lon1_rad)
    z1 = math.sin(p1_lat1_rad)
    x2 = math.cos(p1_lat2_rad) * math.cos(p1_lon2_rad)
    y2 = math.cos(p1_lat2_rad) * math.sin(p1_lon2_rad)
    z2 = math.sin(p1_lat2_rad)
    cx1 = math.cos(p2_lat1_rad) * math.cos(p2_lon1_rad)
    cy1 = math.cos(p2_lat1_rad) * math.sin(p2_lon1_rad)
    cz1 = math.sin(p2_lat1_rad)
    cx2 = math.cos(p2_lat2_rad) * math.cos(p2_lon2_rad)
    cy2 = math.cos(p2_lat2_rad) * math.sin(p2_lon2_rad)
    cz2 = math.sin(p2_lat2_rad)

    # Get normal to planes containing great circles
    # np.cross product of vector to each point from the origin
    N1 = np.cross([x1, y1, z1], [x2, y2, z2])
    N2 = np.cross([cx1, cy1, cz1], [cx2, cy2, cz2])

    # Find line of intersection between two planes
    L = np.cross(N1, N2)

    # Find two intersection points
    X1 = L / np.sqrt(L[0] ** 2 + L[1] ** 2 + L[2] ** 2)
    X2 = -X1
    i_lat1 = math.asin(X1[2]) * 180.0 / np.pi
    i_lon1 = math.atan2(X1[1], X1[0]) * 180.0 / np.pi
    i_lat2 = math.asin(X2[2]) * 180.0 / np.pi
    i_lon2 = math.atan2(X2[1], X2[0]) * 180.0 / np.pi

    return (PPoint(i_lon1, i_lat1), PPoint(i_lon2, i_lat2))


def draw_arc(to, fr, s, corrx=[0, 0], corry=[0, 0], x=None, y=None, inaxcoords=False):
    if y is None:
        mid = intersect(
            (fr, to), (PPoint((to.x + fr.x) / 2, -89), PPoint((to.x + fr.x) / 2, 89))
        )
        if mid[0].x < min(fr.x, to.x) or mid[0].x > max(fr.x, to.x):
            mid = mid[1]
        else:
            mid = mid[0]
    else:
        if x is None:
            mid = PPoint((fr.x + to.x) / 2, y)
        else:
            mid = PPoint(x, y)
    # ax.scatter([mid.x], [mid.y], color='red', zorder=10)
    if not inaxcoords:
        fr = transform(projection, Point(fr.x, fr.y))
    fr = PPoint(fr.x + corrx[1], fr.y + corry[1])
    if not inaxcoords:
        to = transform(projection, Point(to.x, to.y))
    to = PPoint(to.x + corrx[0], to.y + corry[0])
    if not inaxcoords:
        mid = transform(projection, Point(mid.x, mid.y))
    mid = PPoint(mid.x + (corrx[0] + corrx[1]) / 2, mid.y + (corry[0] + corry[1]) / 2)
    perp1 = (fr % to) * s / 4
    perp2 = (fr % (mid + (mid - (fr + to) / 2) * 1.3)) * s / 2
    res = cardinal_curve([to, mid - perp1, fr - perp2])
    w = wedge_to(fr + perp2, fr - perp2, radius=s / 2)
    w.operations[0] = Path.LINETO
    res += w
    w = cardinal_curve([fr + perp2, mid + perp1, to])
    w.operations[0] = Path.LINETO
    res += w
    return res


def bezier_curve_to(a, b, c):
    return PPath([a, b, c], [Path.CURVE4, Path.CURVE4, Path.CURVE4])


# from https://github.com/d3/d3-shape/blob/master/src/curve/cardinal.js
def cardinal_curve(points, k=0.3, first_op=Path.MOVETO):
    p0 = p1 = p2 = None
    res = PPath()

    def point(p):
        return bezier_curve_to(
            PPoint(p1[0] + k * (p2[0] - p0[0]), p1[1] + k * (p2[1] - p0[1])),
            PPoint(p2[0] + k * (p1[0] - p[0]), p2[1] + k * (p1[1] - p[1])),
            PPoint(p2[0], p2[1]),
        )

    for i, p in enumerate(points):
        p = copy.copy(p)
        if i == 0:
            res += PPath([p], [first_op])
        elif i == 1:
            p1 = p
        else:
            res += point(p)
        p0 = p1
        p1 = p2
        p2 = p
    if len(points) == 2:
        res += PPath([p2], [Path.LINETO])
    elif len(points) > 2:
        res += point(p1)
    return res


def wedge_to(fr, to, radius, param1=True, param2=False):
    d = to - fr
    q = d.length()
    if q > 2.0 * radius:
        radius = q / 2
    mid = (fr + to) / 2
    D = math.sqrt(radius ** 2 - (q / 2) ** 2)
    if param1:
        c = PPoint(mid.x - D * d.y / q, mid.y + D * d.x / q)
    else:
        c = PPoint(mid.x + D * d.y / q, mid.y - D * d.x / q)
    theta1 = 180.0 * math.acos(abs(fr.x - c.x) / radius) / math.pi
    if fr.x < c.x:
        theta1 = 180.0 - theta1
    if fr.y < c.y:
        theta1 = -theta1
    theta2 = 180.0 * math.acos(abs(to.x - c.x) / radius) / math.pi
    if to.x < c.x:
        theta2 = 180.0 - theta2
    if to.y < c.y:
        theta2 = -theta2
    if param2:
        arc = Path.arc(theta1, theta2)
    else:
        arc = Path.arc(theta2, theta1)
    return PPath.from_matplotlib(arc) * radius + c


def sgn(a):
    return -1 if a < 0 else 1


max_radius = 1e5  # 7e5

reg_coords = {
    'USA-OTH': (-100, 39),
    'USA': (-100, 40),
    'EUR': (13, 48),
    'CHN': (103, 35),
    'MEX': (-99, 22),
    'CAN': (-105, 58),
    'US.NJ': (-74.405661, 40.058324),
    'US.NY': (-74.217933, 43.299428),
}

for region in list(reg_coords.keys()):
    coord = reg_coords[region]
    coord = transform(world_projection, Point(coord))
    x = coord.x
    y = coord.y
    if region == 'CHN':
        x += chn_offset_x
        y += chn_offset_y
    elif region == 'EUR':
        x += eur_offset_x
        y += eur_offset_y
    reg_coords[region] = (x, y)

trade_flows = pd.read_csv("../data/generated/us_trade_flows_for_concept_figure.csv")
trade_flows = trade_flows[(trade_flows['region_from'].isin(list(reg_coords.keys()))) & (trade_flows['region_to'].isin(list(reg_coords.keys())))]
trade_flows.drop(trade_flows[(~trade_flows['region_from'].isin(['US.NJ', 'US.NY'])) & (~trade_flows['region_to'].isin(['US.NJ', 'US.NY']))].index, inplace=True)
trade_flows.drop((trade_flows[trade_flows['region_from'].isin(['USA-OTH', 'USA'])]).index, inplace=True)
trade_flows.drop((trade_flows[trade_flows['region_to'].isin(['USA-OTH', 'USA'])]).index, inplace=True)
trade_flows.drop(trade_flows[(trade_flows['region_from'].isin(['US.NJ', 'US.NY'])) & (trade_flows['region_to'].isin(['US.NJ', 'US.NY']))].index, inplace=True)
trade_flows.set_index(['region_from', 'region_to'], inplace=True, drop=True)


def move_point(_coord, _vector, _scale=1.0):
    # vec = tuple(c * _scale for c in _vector)
    vec = (_vector[0] * _scale, _vector[1] * _scale)
    # res = tuple(c + vec[i] for c, i in enumerate(_coord))
    res = (_coord[0] + vec[0], _coord[1] + vec[1])
    return res


def get_orthogonal(_p1, _p2):
    # line = tuple(_p1[i] - _p2[i] for i in range(2))
    line = (_p1[0] - _p2[0], _p1[1] - _p2[1])
    res_x = 1
    res_y = (-line[0] * res_x) / line[1]
    res = (res_x / np.sqrt(res_x**2 + res_y**2), res_y / np.sqrt(res_x**2 + res_y**2))
    return res


def get_center(_p1, _p2):
    res = (_p1[0] + (_p2[0] - _p1[0]) / 2, _p1[1] + (_p2[1] - _p1[1]) / 2)
    return res


def get_arc_y(_p1, _p2, _arc_height=0.4e6):
    center = get_center(_p1, _p2)
    orthogonal = get_orthogonal(_p1, _p2)
    res = (move_point(center, orthogonal, _arc_height)[1], move_point(center, orthogonal, -1 * _arc_height)[1])
    return res


inset_corners = {
    'll': (inset_x1, inset_y1),
    'lr': (inset_x2, inset_y1),
    'ul': (inset_x1, inset_y2),
    'ur': (inset_x2, inset_y2),
}

inset_anchor_pts = {
    'll_0': move_point(inset_corners['ll'], get_orthogonal(inset_corners['ll'], inset_corners['ur']), 1e5),
    'll_1': move_point(inset_corners['ll'], get_orthogonal(inset_corners['ll'], inset_corners['ur']), -1e5),
    'lr_0': move_point(inset_corners['lr'], get_orthogonal(inset_corners['lr'], inset_corners['ul']), 1e5),
    'lr_1': move_point(inset_corners['lr'], get_orthogonal(inset_corners['lr'], inset_corners['ul']), -1e5),
    'ur_0': move_point(inset_corners['ur'], get_orthogonal(inset_corners['ur'], inset_corners['ll']), 1e5),
    'ur_1': move_point(inset_corners['ur'], get_orthogonal(inset_corners['ur'], inset_corners['ll']), -1e5),
    'ul_0': move_point(inset_corners['ul'], get_orthogonal(inset_corners['ul'], inset_corners['lr']), 1e5),
    'ul_1': move_point(inset_corners['ul'], get_orthogonal(inset_corners['ul'], inset_corners['lr']), -1e5),
}

inset_anchor_pts['ul_0'] = move_point(inset_anchor_pts['ul_0'], (0, 1), _scale=1e5)
inset_anchor_pts['ur_1'] = move_point(inset_anchor_pts['ur_1'], (1, 0), _scale=2e5)
inset_anchor_pts['lr_1'] = move_point(inset_anchor_pts['lr_1'], (1, 0), _scale=1e5)
inset_anchor_pts['ll_0'] = move_point(inset_anchor_pts['ll_0'], (0, 1), _scale=-1e5)

inset_center = (inset_x1 + (inset_x2 - inset_x1) / 2, inset_y1 + (inset_y2 - inset_y1) / 2)
# inset_anchor_pts = {}
# for corner in inset_corners.keys():
#     inset_anchor_pts[corner+"_0"] = move_point(inset_corners[corner], get_orthogonal(inset_corners[corner], inset_center), 1e5)
#     inset_anchor_pts[corner+"_1"] = move_point(inset_corners[corner], get_orthogonal(inset_corners[corner], inset_center), -1e5)

# reg_anchor_pts = {}
# for label in reg_coords.keys():
#     reg_anchor_pts[label+"_0"] = move_point(reg_coords[label], get_orthogonal(reg_coords[label], inset_center), 1e5)
#     reg_anchor_pts[label+"_1"] = move_point(reg_coords[label], get_orthogonal(reg_coords[label], inset_center), -1e5)

reg_anchor_pts = {
    "CHN_0": move_point(reg_coords["CHN"], get_orthogonal(reg_coords["CHN"], inset_center), 1e5),
    "CHN_1": move_point(reg_coords["CHN"], get_orthogonal(reg_coords["CHN"], inset_center), -1e5),
    "EUR_0": move_point(reg_coords["EUR"], get_orthogonal(reg_coords["EUR"], inset_center), 1e5),
    "EUR_1": move_point(reg_coords["EUR"], get_orthogonal(reg_coords["EUR"], inset_center), -1e5),
    "CAN_0": move_point(reg_coords["CAN"], get_orthogonal(reg_coords["CAN"], inset_center), 1e5),
    "CAN_1": move_point(reg_coords["CAN"], get_orthogonal(reg_coords["CAN"], inset_center), -1e5),
    "MEX_0": move_point(reg_coords["MEX"], get_orthogonal(reg_coords["MEX"], inset_center), 1e5),
    "MEX_1": move_point(reg_coords["MEX"], get_orthogonal(reg_coords["MEX"], inset_center), -1e5),
}

reg_anchor_pts['CHN_0'] = move_point(reg_anchor_pts["CHN_0"], (1, 0), -1e5)
reg_anchor_pts['CHN_1'] = move_point(reg_anchor_pts["CHN_1"], (1, 0), -1e5)
reg_anchor_pts['EUR_0'] = move_point(reg_anchor_pts["EUR_0"], (1, 0), -1e5)
reg_anchor_pts['EUR_1'] = move_point(reg_anchor_pts["EUR_1"], (1, 0), -1e5)

arrows = [
    ('US.NJ', trade_flows.loc[('CHN', 'US.NJ'), 'eora'], PPoint(*move_point(reg_anchor_pts['CHN_1'], get_orthogonal(reg_anchor_pts['CHN_1'], inset_anchor_pts['lr_1']), -5e4)), PPoint(*move_point(inset_anchor_pts['lr_1'], get_orthogonal(reg_anchor_pts['CHN_1'], inset_anchor_pts['lr_1']), -5e4)), get_arc_y(inset_anchor_pts['lr_1'], reg_anchor_pts['CHN_1'])[1] - 5e4),
    ('US.NJ', trade_flows.loc[('US.NJ', 'CHN'), 'eora'], PPoint(*move_point(inset_anchor_pts['lr_1'], get_orthogonal(reg_anchor_pts['CHN_1'], inset_anchor_pts['lr_1']), 5e4)), PPoint(*move_point(reg_anchor_pts['CHN_1'], get_orthogonal(reg_anchor_pts['CHN_1'], inset_anchor_pts['lr_1']), +5e4)), get_arc_y(inset_anchor_pts['lr_1'], reg_anchor_pts['CHN_1'])[1] + 5e4),
    ('US.NY', trade_flows.loc[('CHN', 'US.NY'), 'eora'], PPoint(*move_point(reg_anchor_pts['CHN_0'], get_orthogonal(reg_anchor_pts['CHN_0'], inset_anchor_pts['lr_0']), -5e4)), PPoint(*move_point(inset_anchor_pts['lr_0'], get_orthogonal(reg_anchor_pts['CHN_0'], inset_anchor_pts['lr_0']), -5e4)), get_arc_y(inset_anchor_pts['lr_0'], reg_anchor_pts['CHN_0'])[0] - 5e4),
    ('US.NY', trade_flows.loc[('US.NY', 'CHN'), 'eora'], PPoint(*move_point(inset_anchor_pts['lr_0'], get_orthogonal(reg_anchor_pts['CHN_0'], inset_anchor_pts['lr_0']), 5e4)), PPoint(*move_point(reg_anchor_pts['CHN_0'], get_orthogonal(reg_anchor_pts['CHN_0'], inset_anchor_pts['lr_0']), +5e4)), get_arc_y(inset_anchor_pts['lr_0'], reg_anchor_pts['CHN_0'])[0] + 5e4),

    ('US.NY', trade_flows.loc[('EUR', 'US.NY'), 'eora'], PPoint(*move_point(reg_anchor_pts['EUR_1'], get_orthogonal(reg_anchor_pts['EUR_1'], inset_anchor_pts['ur_1']), -8e4)), PPoint(*move_point(inset_anchor_pts['ur_1'], get_orthogonal(reg_anchor_pts['EUR_1'], inset_anchor_pts['ur_1']), -8e4)), get_arc_y(inset_anchor_pts['ur_1'], reg_anchor_pts['EUR_1'])[1] + 8e4),
    ('US.NY', trade_flows.loc[('US.NY', 'EUR'), 'eora'], PPoint(*move_point(inset_anchor_pts['ur_1'], get_orthogonal(reg_anchor_pts['EUR_1'], inset_anchor_pts['ur_1']), 8e4)), PPoint(*move_point(reg_anchor_pts['EUR_1'], get_orthogonal(reg_anchor_pts['EUR_1'], inset_anchor_pts['ur_1']), +8e4)), get_arc_y(inset_anchor_pts['ur_1'], reg_anchor_pts['EUR_1'])[1] - 8e4),
    ('US.NJ', trade_flows.loc[('EUR', 'US.NJ'), 'eora'], PPoint(*move_point(reg_anchor_pts['EUR_0'], get_orthogonal(reg_anchor_pts['EUR_0'], inset_anchor_pts['ur_0']), -8e4)), PPoint(*move_point(inset_anchor_pts['ur_0'], get_orthogonal(reg_anchor_pts['EUR_0'], inset_anchor_pts['ur_0']), -8e4)), get_arc_y(inset_anchor_pts['ur_0'], reg_anchor_pts['EUR_0'])[0] + 8e4),
    ('US.NJ', trade_flows.loc[('US.NJ', 'EUR'), 'eora'], PPoint(*move_point(inset_anchor_pts['ur_0'], get_orthogonal(reg_anchor_pts['EUR_0'], inset_anchor_pts['ur_0']), 8e4)), PPoint(*move_point(reg_anchor_pts['EUR_0'], get_orthogonal(reg_anchor_pts['EUR_0'], inset_anchor_pts['ur_0']), +8e4)), get_arc_y(inset_anchor_pts['ur_0'], reg_anchor_pts['EUR_0'])[0] - 8e4),

    ('US.NJ', trade_flows.loc[('CAN', 'US.NJ'), 'eora'], PPoint(*move_point(reg_anchor_pts['CAN_1'], get_orthogonal(reg_anchor_pts['CAN_1'], inset_anchor_pts['ul_1']), -5e4)), PPoint(*move_point(inset_anchor_pts['ul_1'], get_orthogonal(reg_anchor_pts['CAN_1'], inset_anchor_pts['ul_1']), -5e4)), get_arc_y(inset_anchor_pts['ul_1'], reg_anchor_pts['CAN_1'])[1] - 5e4),
    ('US.NJ', trade_flows.loc[('US.NJ', 'CAN'), 'eora'], PPoint(*move_point(inset_anchor_pts['ul_1'], get_orthogonal(reg_anchor_pts['CAN_1'], inset_anchor_pts['ul_1']), 5e4)), PPoint(*move_point(reg_anchor_pts['CAN_1'], get_orthogonal(reg_anchor_pts['CAN_1'], inset_anchor_pts['ul_1']), +5e4)), get_arc_y(inset_anchor_pts['ul_1'], reg_anchor_pts['CAN_1'])[1] + 5e4),
    ('US.NY', trade_flows.loc[('CAN', 'US.NY'), 'eora'], PPoint(*move_point(reg_anchor_pts['CAN_0'], get_orthogonal(reg_anchor_pts['CAN_0'], inset_anchor_pts['ul_0']), -5e4)), PPoint(*move_point(inset_anchor_pts['ul_0'], get_orthogonal(reg_anchor_pts['CAN_0'], inset_anchor_pts['ul_0']), -5e4)), get_arc_y(inset_anchor_pts['ul_0'], reg_anchor_pts['CAN_0'])[0] - 5e4),
    ('US.NY', trade_flows.loc[('US.NY', 'CAN'), 'eora'], PPoint(*move_point(inset_anchor_pts['ul_0'], get_orthogonal(reg_anchor_pts['CAN_0'], inset_anchor_pts['ul_0']), 5e4)), PPoint(*move_point(reg_anchor_pts['CAN_0'], get_orthogonal(reg_anchor_pts['CAN_0'], inset_anchor_pts['ul_0']), +5e4)), get_arc_y(inset_anchor_pts['ul_0'], reg_anchor_pts['CAN_0'])[0] + 5e4),

    ('US.NY', trade_flows.loc[('MEX', 'US.NY'), 'eora'], PPoint(*move_point(reg_anchor_pts['MEX_1'], get_orthogonal(reg_anchor_pts['MEX_1'], inset_anchor_pts['ll_1']), -5e4)), PPoint(*move_point(inset_anchor_pts['ll_1'], get_orthogonal(reg_anchor_pts['MEX_1'], inset_anchor_pts['ll_1']), -5e4)), get_arc_y(inset_anchor_pts['ll_1'], reg_anchor_pts['MEX_1'])[1] + 5e4),
    ('US.NY', trade_flows.loc[('US.NY', 'MEX'), 'eora'], PPoint(*move_point(inset_anchor_pts['ll_1'], get_orthogonal(reg_anchor_pts['MEX_1'], inset_anchor_pts['ll_1']), 5e4)), PPoint(*move_point(reg_anchor_pts['MEX_1'], get_orthogonal(reg_anchor_pts['MEX_1'], inset_anchor_pts['ll_1']), +5e4)), get_arc_y(inset_anchor_pts['ll_1'], reg_anchor_pts['MEX_1'])[1] - 5e4),
    ('US.NJ', trade_flows.loc[('MEX', 'US.NJ'), 'eora'], PPoint(*move_point(reg_anchor_pts['MEX_0'], get_orthogonal(reg_anchor_pts['MEX_0'], inset_anchor_pts['ll_0']), -5e4)), PPoint(*move_point(inset_anchor_pts['ll_0'], get_orthogonal(reg_anchor_pts['MEX_0'], inset_anchor_pts['ll_0']), -5e4)), get_arc_y(inset_anchor_pts['ll_0'], reg_anchor_pts['MEX_0'])[0] + 5e4),
    ('US.NJ', trade_flows.loc[('US.NJ', 'MEX'), 'eora'], PPoint(*move_point(inset_anchor_pts['ll_0'], get_orthogonal(reg_anchor_pts['MEX_0'], inset_anchor_pts['ll_0']), 5e4)), PPoint(*move_point(reg_anchor_pts['MEX_0'], get_orthogonal(reg_anchor_pts['MEX_0'], inset_anchor_pts['ll_0']), +5e4)), get_arc_y(inset_anchor_pts['ll_0'], reg_anchor_pts['MEX_0'])[0] - 5e4),

]

norm = Normalize(vmin=0, vmax=trade_flows['eora'].max())


reg_la = {
    'CHN': {
        'h': 'left',
        'v': 'center',
    },
    'EUR': {
        'h': 'left',
        'v': 'center',
    },
    'MEX': {
        'h': 'right',
        'v': 'top',
    },
    'USA-OTH': {
        'h': 'center',
        'v': 'center',
    },
    'USA': {
        'h': 'center',
        'v': 'center',
    },
    'CAN': {
        'h': 'right',
        'v': 'bottom',
    },
}
for label in ['USA', 'EUR', 'CHN', 'MEX', 'CAN']:
# for label in ['CHN']:
    ax.annotate(label, (reg_coords[label][0], reg_coords[label][1]), ha=reg_la[label]['h'], va=reg_la[label]['v'], fontsize=FSIZE_SMALL)

for a in arrows:
    d1 = a[1]
    corrx = [0, 0]
    corry = [0, 0]
    ax.add_patch(
        PathPatch(
            draw_arc(
                a[2],
                a[3],
                2 * max_radius * norm(d1),
                corrx=corrx,
                corry=corry,
                y=a[4],
                inaxcoords=True
            ).to_matplotlib(),
            lw=0.1,
            fc=us_state_colors[a[0]],
            ec='none'
        )
    )

row_distance = 0.35e6

row = ax.get_ylim()[0] + 0.2e6
ax.text(ax.get_xlim()[0], row, 'Importing region', ha='left', va='center')
ax.text(-1.0e7, row, 'Exporting region', ha='left', va='center')
ax.add_patch(PathPatch(draw_arc(PPoint(-1.01e7, row), PPoint(-1.06e7, row), 2*max_radius, y=ax.get_ylim()[0] + 0.3e6, inaxcoords=True).to_matplotlib(), lw=0.1, fc='k', ec='none'))

row = row + row_distance
circle_x = ax.get_xlim()[0] + 0.2e6
circle_text_dist = ax.get_xlim()[0] + 0.5e6 - (circle_x + max_radius * norm(50e6) / 2)
ax.add_patch(mpl.patches.Circle((circle_x, row), max_radius * norm(50e6), color='k'))
ax.text(circle_x + (max_radius * norm(50e6) / 2 + circle_text_dist), row, '$50bn', ha='left', va='center')
circle_x = circle_x + 1.5e6 - max_radius * norm(30e6) / 2
ax.add_patch(mpl.patches.Circle((circle_x, row), max_radius * norm(30e6), color='k'))
ax.text(circle_x + max_radius * norm(30e6) / 2 + circle_text_dist, row, '$30bn', ha='left', va='center')
circle_x = circle_x + 1.5e6 - max_radius * norm(5e6) / 2
ax.add_patch(mpl.patches.Circle((circle_x, row), max_radius * norm(5e6), color='k'))
ax.text(circle_x + max_radius * norm(5e6) / 2 + circle_text_dist, row, '$5bn', ha='left', va='center')

# ax.add_patch(mpl.patches.Circle((ax.get_xlim()[0]+0.5e6, row), max_radius * norm(50e6), color='k'))
# ax.text(-1.0e7, row, 'Exporting Region', ha='left', va='center')

row = row + row_distance
ax.plot([ax.get_xlim()[0], ax.get_xlim()[0] + 0.45e6], [row, row], linestyle='dotted', color='k')
ax.text(ax.get_xlim()[0] + 0.5e6, row, 'Sandy track', ha='left', va='center')

row = row + row_distance
ax.add_patch(mpl.patches.Rectangle((ax.get_xlim()[0], row - 0.1e6), 0.4e6, 0.2e6, facecolor='gray', edgecolor='none'))
ax.text(ax.get_xlim()[0] + 0.5e6, row, 'US states with minor Sandy impact', va='center')

row = row + row_distance
ax.add_patch(mpl.patches.Rectangle((ax.get_xlim()[0], row - 0.1e6), 0.4e6, 0.2e6, facecolor='white', edgecolor='k', hatch='//////////', linewidth=0.1))
ax.text(ax.get_xlim()[0] + 0.5e6, row, 'High water marks', va='center')

row = row + row_distance
ax.add_patch(mpl.patches.Rectangle((ax.get_xlim()[0], row - 0.1e6), 0.4e6, 0.2e6, facecolor=us_state_colors['US.NY'], edgecolor='none'))
ax.text(ax.get_xlim()[0] + 0.5e6, row, 'NY', va='center')

row = row + row_distance
ax.add_patch(mpl.patches.Rectangle((ax.get_xlim()[0], row - 0.1e6), 0.4e6, 0.2e6, facecolor=us_state_colors['US.NJ'], edgecolor='none'))
ax.text(ax.get_xlim()[0] + 0.5e6, row, 'NJ', va='center')

# ax.axis('scaled')
plt.tight_layout()
fig.savefig("../figures/conceptual_fig_wo_USA-OTH.pdf", dpi=300)
plt.show()