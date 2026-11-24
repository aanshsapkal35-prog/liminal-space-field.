#!/usr/bin/env python3
"""
liminal_space_field.py  --  720x1280 @ 30 fps, 24 s seamless loop.

A night field of daisies under a moon that slips in and out of the clouds,
painted small with numpy, then read back as ASCII, with the source scrolling below.

Re-created from a screen recording.  The algorithm for the backdrop, the wind, the
noise, the blades, the daisies, the falling stars and the sparks follows the source
that is visible in the recording line for line; the constants that the recording
never shows (colours, camera, the cloud ceiling) were measured from its frames.

    python3 liminal_space_field.py                 # -> liminal_space_field.mp4
    python3 liminal_space_field.py --frames 0 1 2  # PNG test frames
"""
import argparse, json, math, os, subprocess, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- layout
OUT_W, OUT_H = 720, 1280
W, H = 720, 864                  # the scene (the painting + letters)
HOR = H // 2                     # horizon row
FPS, LOOP = 30, 24.0
CELL_W, CELL_H = 8, 12           # one letter
COLS, ROWS = W // CELL_W, H // CELL_H           # 90 x 72
TW, TH = W // 4, H // 4          # the painting is made at a quarter size
N_SKY = HOR // 4

BEAT = 2 * math.pi / 8.0         # one gust-beat = 8 s ; 8*BEAT -> 1 Hz
TILE = 512
STRIP = 5.5                      # one strip of field; the loop moves the camera one strip
DRIFT = STRIP / LOOP
NEAR = 1.55
F = 923.0                        # focal length in px   (0.39 = 360 / F)
CAMH = 0.75                      # camera height

N_DAISIES, DAISY_TILES, DAISY_HALF = 900, 8, 18.0
N_BLADES, BLADE_TILES, BLADE_HALF = 3600, 3, 8.0
N_PETALS, N_FACES = 13, 6        # petals on a daisy; ways a daisy can face
DAISY_SIZES = 5.0 * 1.15 ** np.arange(27)       # the sizes a near daisy is drawn at
T_MOON = (1.03, 3.05, 13.45, 16.35)
STARS = [(-0.10, 0.058, 0.046, 0.85), (4.45, -0.010, 0.150, 0.85), (8.20, 0.190, 0.026, 0.95),
         (12.00, 0.054, 0.096, 0.90), (15.45, 0.712, 0.023, 0.75)]
MOON_XY = (490.0, 172.0)
MOON_R = 26.0 / H

# colours (0..255)
ZENITH = np.array([8, 24, 104.]);     SKYLINE = np.array([42, 84, 236.])
CLOUD_DARK = np.array([4, 10, 52.]);   CLOUD_LIT = np.array([112, 164, 255.])
MOON = np.array([240, 244, 255.]);    HAZE = np.array([60, 98, 180.])
FIELD_FAR = np.array([44, 72, 104.]); FIELD_NEAR = np.array([12, 42, 36.])
FIELD_MOON = np.array([17, 62, 57.]); GUST = np.array([50, 100, 86.])
TREES = np.array([6, 14, 36.])
BLADE_DARK = np.array([9, 36, 33.]); BLADE_LIGHT = np.array([34, 86, 74.])
PETAL_FAR = np.array([140, 168, 205.]); PETAL_DUSK = np.array([108, 134, 232.])
PETAL_MOON = np.array([232, 238, 250.])
FIREFLY = np.array([190, 255, 110.])


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# ----------------------------------------------------------------------------- noise
def make_noise(n, beta, seed):
    # soft noise that tiles, every value from 0 to 1 equally often
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n)
    k = np.hypot(*np.meshgrid(f, f))
    k[0, 0] = 1.0
    img = np.fft.ifft2((rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / k ** beta).real
    rank = np.empty(n * n)
    rank[np.argsort(img.ravel())] = np.linspace(0, 1, n * n)
    return rank.reshape(n, n)


NOISE_A, NOISE_B = make_noise(TILE, 1.75, 4), make_noise(TILE, 1.55, 9)


def tex(noise, u, v):
    return noise[v.astype(int) & (TILE - 1), u.astype(int) & (TILE - 1)]


def wind(x, z, t):
    # gusts crossing the field, 0..1; in time with the loop
    return 0.5 + 0.5 * np.sin(0.55 * x + 0.35 * z - 8 * BEAT * t + 1.4 * np.sin(0.13 * x + 0.21 * z - 2 * BEAT * t))


def moonlight(t):
    # 0 the moon behind cloud, 1 out: covered at first, soon out, covered again at the end
    return float(smooth((t - T_MOON[0]) / (T_MOON[1] - T_MOON[0])) *
                 (1 - smooth((t - T_MOON[2]) / (T_MOON[3] - T_MOON[2]))))


# ----------------------------------------------------------------------------- scene
class Field:
    def __init__(self):
        self.w, self.h, self.tw, self.th, self.n_sky = W, H, TW, TH, N_SKY
        rng = np.random.default_rng(7)
        # the small painting's pixel centres
        cx = (np.arange(TW) + 0.5) * 4.0
        ry = (np.arange(TH) + 0.5) * 4.0
        px, py = np.meshgrid(cx, ry)
        s_px, s_py = px[:N_SKY], py[:N_SKY]
        self.sky_up = (HOR - s_py) / HOR
        self.moon_far = np.hypot(s_px - MOON_XY[0], s_py - MOON_XY[1]) / H
        # a ceiling of cloud seen from below
        d = np.maximum(self.sky_up, 0.02)
        self.cloud_u = 105.0 * (s_px - W / 2) / HOR / d ** 0.25
        self.cloud_v = 150.0 / d ** 0.85
        g_px, g_py = px[N_SKY:], py[N_SKY:]
        dy = g_py - HOR
        self.ground_z = F * CAMH / dy
        self.ground_x = (g_px - W / 2) * self.ground_z / F
        # blades
        nb = N_BLADES
        self.BLADE = dict(x=rng.uniform(-BLADE_HALF, BLADE_HALF, nb), z=rng.uniform(0, STRIP, nb),
                          tall=rng.uniform(0.07, 0.26, nb), lean=rng.normal(0, 0.06, nb),
                          rank=rng.uniform(0, 1, nb), tone=rng.uniform(0, 1, nb), seed=rng.uniform(0, 6.283, nb))
        nd = N_DAISIES
        r1 = np.random.default_rng(264)
        dx, dz, drank = r1.uniform(-DAISY_HALF, DAISY_HALF, nd), r1.uniform(0, STRIP, nd), r1.uniform(0, 1, nd)
        self.DAISY = dict(x=dx, z=dz, tall=rng.uniform(0.20, 0.40, nd), size=rng.uniform(0.082, 0.135, nd),
                          face=rng.integers(0, N_FACES, nd), rank=drank, seed=rng.uniform(0, 6.283, nd))
        self.sprites = {}
        self.build_trees()
        self.fire = dict(x=rng.uniform(-3.2, 3.2, 24), z=rng.uniform(1.8, 8.5, 24),
                         y=rng.uniform(0.15, 0.9, 24), ph=rng.uniform(0, 6.283, 24),
                         fq=rng.uniform(0.6, 1.4, 24), thr=rng.uniform(0, 1, 24))
        self.halo = self.make_halo()
        self.hash = rng.integers(0, 1 << 20, (ROWS, COLS))
        self.make_glyphs()
        self.tinted = {}

    # ---- the tree line
    def build_trees(self):
        rng = np.random.default_rng(11)
        self.trees = []
        for layer, (n, hmin, hmax) in enumerate(((70, 8, 30), (52, 14, 47))):
            for _ in range(n):
                cx = rng.uniform(-10, W + 10)
                hh = rng.uniform(hmin, hmax) ** 1.0
                self.trees.append((cx, hh, hh * rng.uniform(0.18, 0.28)))
        self.trees.sort(key=lambda a: a[1])

    # ---- halo sprite (soft point of light)
    def make_halo(self):
        n = 64
        y, x = np.mgrid[0:n, 0:n]
        r = np.hypot(x - n / 2 + 0.5, y - n / 2 + 0.5) / (n / 2)
        a = np.clip(1 - r, 0, 1) ** 2.2
        return a.astype(np.float32)

    def spark(self, x, y, size, rgb):
        # a soft point of light
        key = (int(size / 4), tuple((np.asarray(rgb) // 16).astype(int).tolist()))
        if key not in self.tinted:
            s = key[0] * 4 + 4
            h_ = cv2.resize(self.halo, (s, s), interpolation=cv2.INTER_AREA)
            tint = np.array([min(255, c * 16 + 8) for c in key[1]], np.float32)
            self.tinted[key] = h_[..., None] * tint[None, None, :] / 255.0
        pic = self.tinted[key]
        s = pic.shape[0]
        x0, y0 = int(round(x - s / 2)), int(round(y - s / 2))
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + s, W), min(y0 + s, H)
        if xb <= xa or yb <= ya:
            return
        self.scene[ya:yb, xa:xb] += pic[ya - y0:yb - y0, xa - x0:xb - x0] * 255.0
        self.sparks.append((x, y, float(np.max(rgb)) / 255.0))

    # ---- sky and ground: painted small, with numpy
    def backdrop(self, t, light):
        n, w, h = self.n_sky, self.w, self.h
        img = np.empty((self.th, self.tw, 3), np.float32)
        # cloud: two layers of noise sliding at their own speed over a ceiling seen from below
        slide = TILE * t / LOOP
        u, v = self.cloud_u, self.cloud_v
        dense = 0.68 * tex(NOISE_A, u + slide, v + 2 * slide) + 0.32 * tex(NOISE_B, 2 * u - slide, 2 * v + slide)
        toward = 0.68 * tex(NOISE_A, u + slide + 16, v + 2 * slide - 16) + 0.32 * tex(NOISE_B, 2 * u - slide + 16, 2 * v + slide - 16)
        gap = np.exp(-(self.moon_far / 0.13) ** 2)              # round the moon it is thin
        cover = smooth((dense - 0.40 - gap * (0.62 * light - 0.30 * (1 - light))) / 0.22)
        lit = np.clip(0.42 + 4.5 * (dense - toward), 0, 1) * (0.55 + 0.45 * light) + 0.0
        lit = np.clip(lit + 0.9 * light * np.exp(-(self.moon_far / 0.20) ** 2), 0, 1)
        self.cover, self.lit = cover, lit
        clear = ZENITH + (SKYLINE - ZENITH) * ((1 - self.sky_up) ** 1.6)[..., None]
        clear = clear + (MOON - clear) * (0.45 * light * np.exp(-(self.moon_far / 0.18) ** 2))[..., None]
        cloud = CLOUD_DARK + (CLOUD_LIT - CLOUD_DARK) * lit[..., None]
        sky = clear + (cloud - clear) * cover[..., None]
        disc = np.clip((MOON_R - self.moon_far) * h / 2.5, 0, 1) * (1 - 0.94 * cover)
        sky = sky + (MOON - sky) * disc[..., None]
        img[:n] = sky + (HAZE - sky) * (0.55 * np.exp(-self.sky_up * 8))[..., None]
        # the field: far off it is haze; near, grass, mottled, with the gusts and the moon
        x, z = self.ground_x, self.ground_z
        zw = z + DRIFT * t
        near = np.exp(-z / 30.0)
        mottle = tex(NOISE_A, x * 40, zw * (TILE / STRIP))        # the ground slides under you
        shadow = tex(NOISE_B, x * 3 + slide, z * 3 + 2 * slide)
        gust = wind(x, z, t) ** 2
        tone = (0.72 + 0.50 * mottle) * (0.86 + 0.28 * shadow * light) * (0.94 + 0.24 * gust)
        field = (FIELD_FAR + (FIELD_NEAR + (FIELD_MOON - FIELD_NEAR) * light - FIELD_FAR) * near[..., None]) * tone[..., None]
        img[n:] = field + (GUST - field) * (0.13 * gust * near)[..., None]
        small = np.clip(img, 0, 255).astype(np.uint8)
        self.scene = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)

    def place(self, x, y, z):
        return W / 2 + F * x / z, HOR + F * (CAMH - y) / z

    # ---- the daisy sprites : size x face, drawn once
    def daisy_sprite(self, si, fi):
        key = (si, fi)
        if key in self.sprites:
            return self.sprites[key]
        S = max(int(round(DAISY_SIZES[si])), 3)
        ss = 4 if S < 40 else 2
        N = S * ss
        rng = np.random.default_rng(100 + fi)
        tilt = 0.45 + 0.55 * (fi / (N_FACES - 1)) ** 0.7        # how open the daisy faces us
        rot = rng.uniform(0, 6.283)
        petals = np.zeros((N, N), np.float32)
        shade = np.zeros((N, N, 3), np.float32)
        c = N / 2 - 0.5
        yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
        xx -= c; yy -= c
        yy = yy / tilt
        rad = np.hypot(xx, yy) / (N / 2)
        ang = np.arctan2(yy, xx) - rot
        k = N_PETALS
        # petal lobes along the angle, long and narrow
        lobe = np.cos(k * ang / 1.0) * 0.5 + 0.5
        edge = 0.62 + 0.38 * lobe ** 1.6
        a = np.clip((edge - rad) * N / 2 / 1.5, 0, 1) * (rad > 0.12)
        petals = a
        # petal veins : darker toward the centre, lighter at the tips
        v = 0.78 + 0.22 * np.clip((rad - 0.18) / 0.5, 0, 1)
        v = v * (0.9 + 0.1 * np.cos(k * ang))
        centre = np.clip((0.215 - rad) * N / 2 / 1.5, 0, 1)
        sprite = np.zeros((N, N, 4), np.float32)
        sprite[..., 0] = v          # petal tone (multiplies the petal colour)
        sprite[..., 1] = centre     # centre mask
        sprite[..., 2] = rad
        sprite[..., 3] = np.maximum(petals, centre)
        if ss > 1:
            sprite = cv2.resize(sprite, (S, S), interpolation=cv2.INTER_AREA)
        if S >= 12:
            sprite = cv2.GaussianBlur(sprite, (0, 0), S / 70.0)
        self.sprites[key] = sprite
        return sprite

    # ---- everything that grows in the field
    def plants(self, t, light):
        scene = self.scene
        cz = DRIFT * t
        # blades: a strip of them, laid end to end; fewer far off
        B = self.BLADE
        tile = np.repeat(np.arange(BLADE_TILES), N_BLADES)
        z = np.tile((B["z"] - cz) % STRIP, BLADE_TILES) + tile * STRIP + NEAR
        x, tall, lean = (np.tile(B[k], BLADE_TILES) for k in ("x", "tall", "lean"))
        rank, tone, seed = (np.tile(B[k], BLADE_TILES) for k in ("rank", "tone", "seed"))
        end = BLADE_TILES * STRIP + NEAR
        keep = (np.abs(x) < 0.39 * z + 0.3) & (rank < np.clip((5.5 / z) ** 1.1, 0.30, 1.0))
        x, z, tall, lean, tone, seed = x[keep], z[keep], tall[keep], lean[keep], tone[keep], seed[keep]
        gust = wind(x, z, t) ** 2
        bend = (lean + 0.55 * gust + 0.05 * np.sin(6 * BEAT * t + seed)) * tall
        x0, y0 = self.place(x, 0.0, z)
        x1, y1 = self.place(x + 0.35 * bend, 0.55 * tall, z)
        x2, y2 = self.place(x + bend, tall * (1 - 0.15 * gust), z)
        fade = smooth((end - z) / 5.0) * np.exp(-z / 22.0)      # they melt into the haze
        rgb = (BLADE_DARK + (BLADE_LIGHT - BLADE_DARK) * (tone * (0.35 + 0.65 * gust) * (0.55 + 0.45 * light))[:, None])
        base = FIELD_FAR + (FIELD_NEAR - FIELD_FAR) * np.exp(-z / 30.0)[:, None]
        rgb = np.clip(base + (rgb - base) * fade[:, None], 0, 255).astype(int)
        wide = np.clip(F * 0.010 / z, 1, 6).astype(int)
        order = np.argsort(-z)
        img = np.ascontiguousarray(np.clip(scene, 0, 255).astype(np.uint8))
        for i in order.tolist():
            c = rgb[i].tolist()
            p0 = (int(x0[i]), int(y0[i])); p1 = (int(x1[i]), int(y1[i])); p2 = (int(x2[i]), int(y2[i]))
            if p0[1] < HOR + 1:
                continue
            cv2.polylines(img, [np.array([p0, p1, p2], np.int32)], False, (c[0], c[1], c[2]), int(wide[i]), cv2.LINE_AA)
        self.scene = img.astype(np.float32)

        # daisies: each on its stalk, nodding in the gust that is passing; more of them far off
        D = self.DAISY
        tile = np.repeat(np.arange(DAISY_TILES), N_DAISIES)
        z = np.tile((D["z"] - cz) % STRIP, DAISY_TILES) + tile * STRIP + NEAR
        x, tall, size, face, rank, seed = (np.tile(D[k], DAISY_TILES) for k in ("x", "tall", "size", "face", "rank", "seed"))
        keep = (np.abs(x) < 0.39 * z + 0.25) & (rank < np.clip(7.0 / z, 0.30, 1.0))
        x, z, tall, size, face, seed = x[keep], z[keep], tall[keep], size[keep], face[keep], seed[keep]
        self.n_daisies = int(keep.sum())
        gust = wind(x, z, t) ** 2
        nod = 0.03 * np.sin(4 * BEAT * t + seed) + 0.13 * gust
        hx, hy = self.place(x + nod, tall - 0.04 * gust, z)
        fx, fy = self.place(x, 0.0, z)
        r = F * size / z
        near = np.round(np.exp(-z / 16.0) * 6)[:, None] / 6          # in a few steps, so the sprites cache
        petal = PETAL_FAR + ((PETAL_DUSK + (PETAL_MOON - PETAL_DUSK) * round(light * 6) / 6) - PETAL_FAR) * near
        self.heads = (hx, hy, r, petal)
        order = np.argsort(-z)
        img = self.scene
        stalk = np.array([26, 80, 64.])
        for i in order.tolist():
            if hy[i] < HOR - 4 or r[i] < 1.0:
                continue
            col = petal[i]
            if r[i] < 4.5:
                px, py = int(hx[i]), int(hy[i])
                if 0 <= px < W and 0 <= py < H:
                    img[py, px] = img[py, px] * 0.35 + col * 0.65
                continue
            fd = float(np.exp(-z[i] / 22.0))
            sc = stalk * (0.6 + 0.4 * light) + (FIELD_FAR - stalk) * (1 - fd) * 0.7
            cv2.line(img, (int(fx[i]), int(fy[i])), (int(hx[i]), int(hy[i])), tuple(float(c) for c in sc),
                     max(1, int(r[i] / 22)), cv2.LINE_AA)
            si = int(np.clip(round(math.log(max(r[i], 5.0) / 5.0) / math.log(1.15)), 0, 26))
            sp = self.daisy_sprite(si, int(face[i]))
            S = sp.shape[0]
            x0_, y0_ = int(round(hx[i] - S / 2)), int(round(hy[i] - S / 2))
            xa, ya, xb, yb = max(x0_, 0), max(y0_, 0), min(x0_ + S, W), min(y0_ + S, H)
            if xb <= xa or yb <= ya:
                continue
            s_ = sp[ya - y0_:yb - y0_, xa - x0_:xb - x0_]
            a = s_[..., 3:4]
            body = col[None, None, :] * s_[..., 0:1]
            body = body * (0.82 + 0.35 * (1 - s_[..., 2:3])) if False else body
            cen = np.array([236, 204, 62.])[None, None, :] * (0.75 + 0.25 * light) * (1.05 - 0.35 * s_[..., 2:3] * 4)
            colr = body * (1 - s_[..., 1:2]) + cen * s_[..., 1:2]
            dst = img[ya:yb, xa:xb]
            img[ya:yb, xa:xb] = dst * (1 - a) + colr * a
        self.scene = img

    # ---- fireflies and falling stars
    def lights(self, t, light):
        self.sparks = []
        env = smooth((t - 4.2) / 5.0) * (1 - smooth((t - 18.2) / 1.8))
        Fz = self.fire
        cz = DRIFT * t
        zz = (Fz["z"] - cz * 0.0)
        # they hang about, drifting
        fx_ = Fz["x"] + 0.35 * np.sin(0.7 * t + Fz["ph"]) + 0.12 * np.sin(2.3 * t * Fz["fq"])
        fy_ = Fz["y"] + 0.10 * np.sin(1.1 * t + Fz["ph"] * 2)
        fz_ = Fz["z"] + 0.4 * np.sin(0.5 * t + Fz["ph"] * 3)
        blink = 0.5 + 0.5 * np.sin(1.6 * Fz["fq"] * t + Fz["ph"])
        glow = env * (0.30 + 0.70 * blink) * (Fz["thr"] < env * 1.2 + 0.0)
        sx, sy = self.place(fx_, fy_, fz_)
        self.fireflies = 0
        for a, b, g, zi in zip(sx.tolist(), sy.tolist(), glow.tolist(), fz_.tolist()):
            if g > 0.08:
                self.spark(a, b, float(np.clip(F * 0.20 / zi, 26, 84)), FIREFLY * g)
                self.fireflies += 1
        step = np.array([0.34, 0.15]) * H * 0.024              # a falling star: which way it goes
        for t0, across, down, far in STARS:
            age = (t - t0) / 0.8
            if 0 <= age < 1:
                a = np.array([across * W, down * H]) + step / 0.024 * far * age
                tail = np.zeros((H, W, 3), np.uint8)
                for k in range(14):                              # head first, the tail fading behind
                    c = tuple(int(v * (1 - age) ** 0.5 * (1 - k / 14)) for v in (235, 240, 255))
                    cv2.line(tail, tuple(int(round(v)) for v in (a - step * k)),
                             tuple(int(round(v)) for v in (a - step * (k + 1))), c, 3 if k < 5 else 1, cv2.LINE_AA)
                self.scene = np.clip(self.scene + tail.astype(np.float32), 0, 255)
                self.spark(float(a[0]), float(a[1]), 34.0, np.array([235, 240, 255]) * (1 - age) ** 0.5)

    # ---- glyphs for the letters
    def make_glyphs(self):
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
        chars = "".join(sorted(set(" .:;,'`-_=+*oO0@#%&$8QBbdpqgyvVwWYy1iIl|/\\!?(){}[]<>^~")))
        self.chars = chars
        atlas = np.zeros((len(chars), CELL_H, CELL_W), np.float32)
        for i, ch in enumerate(chars):
            im = Image.new("L", (CELL_W, CELL_H), 0)
            ImageDraw.Draw(im).text((0, -1), ch, font=font, fill=255)
            atlas[i] = np.asarray(im, np.float32) / 255.0
        self.atlas = atlas
        self.ci = {c: i for i, c in enumerate(chars)}

    def pick(self, s, idx):
        a = np.array([self.ci[c] for c in s])
        return a[idx % len(a)]

    # ---- the letters: one for every cell, chosen by what the painting shows there
    def letters(self, t, cell):
        cols, rows, h_ = COLS, ROWS, self.hash
        lum = cell @ np.array([0.30, 0.55, 0.15], np.float32)
        slow, quick = (h_ + int(t * 1.8)) % 12, (h_ + (t * 4.2 + h_ % 11).astype(int)) % 12
        r_, g_, b_ = cell[..., 0], cell[..., 1], cell[..., 2]
        row = np.arange(rows)[:, None] * np.ones((1, cols), int)
        sky = row < (HOR // CELL_H) - 3
        idx = np.zeros((rows, cols), int)
        alpha = np.zeros((rows, cols), np.float32)
        # sky : dim dots over the clear sky, heavy letters where the cloud is lit
        s1 = sky & (lum > 52)
        idx[s1] = self.pick(".:.", slow)[s1]; alpha[s1] = 0.35
        s2 = sky & (lum > 95)
        idx[s2] = self.pick("o0ob;:", slow)[s2]; alpha[s2] = 0.62
        s3 = sky & (lum > 125)
        idx[s3] = self.pick("8$Q&#%B@8$0", slow)[s3]; alpha[s3] = 0.82
        # the horizon band of haze: stars
        hz = (row >= (HOR // CELL_H) - 1) & (row < (HOR // CELL_H) + 5)
        s4 = hz & (lum > 74)
        idx[s4] = self.ci["*"]; alpha[s4] = 0.6
        # the field : slim tall letters
        gnd = row >= (HOR // CELL_H) + 0
        f1 = gnd & (lum > 30)
        idx[f1] = self.pick("yvVwW1i|/lIYv", slow)[f1]; alpha[f1] = 0.40
        # daisies : petals are rings, the middle an @
        white = (b_ > 0.62 * r_ + 40) & (np.minimum(np.minimum(r_, g_), b_) > 0.52 * np.maximum(np.maximum(r_, g_), b_)) & (lum > 92)
        yellow = (r_ > 140) & (g_ > 118) & (b_ < 0.75 * g_)
        p1 = gnd & white
        idx[p1] = self.pick("OoOo0", quick)[p1]; alpha[p1] = 0.85
        p2 = gnd & yellow
        idx[p2] = self.ci["@"]; alpha[p2] = 0.95
        # sparks (fireflies, star heads) : a bright dot
        for (x, y, g) in self.sparks:
            c, r = int(x // CELL_W), int(y // CELL_H)
            if 0 <= c < cols and 0 <= r < rows and g > 0.3:
                idx[r, c] = self.ci["*"] if g > 0.7 else self.ci["o"]; alpha[r, c] = 0.9
        self.n_letters = int((idx != self.ci[" "]).sum() - ((alpha == 0) & (idx == 0)).sum())
        self.n_letters = int((alpha > 0).sum())
        # colour of a letter : the painting there, pushed brighter
        col = np.clip(cell * 1.22 + np.array([20, 38, 40.]), 0, 255)
        glow = np.clip(cell * 1.0 + 70, 0, 255)
        col = np.where(sky[..., None], np.clip(cell * 1.18 + np.array([16, 30, 52.]), 0, 255), col)
        col = np.where(p1[..., None] | p2[..., None], np.where(p2[..., None], np.array([250, 235, 120.]), np.clip(cell * 0.95 + 20, 0, 255)), col)
        col = np.where(s4[..., None], np.array([205, 225, 245.]), col)
        g = self.atlas[idx]                                       # rows, cols, 12, 8
        g = g.transpose(0, 2, 1, 3).reshape(rows * CELL_H, cols * CELL_W)
        a = np.repeat(np.repeat(alpha, CELL_H, 0), CELL_W, 1) * g
        c = np.repeat(np.repeat(col, CELL_H, 0), CELL_W, 1)
        self.scene = self.scene * (1 - a[..., None]) + c * a[..., None]


# ----------------------------------------------------------------------------- the code panel
class Panel:
    PANEL_Y = H
    BG = (7, 11, 28); HEAD = (10, 18, 44); BAR = (21, 28, 58)
    TEXT = (176, 182, 214); KW = (196, 146, 190); FN = (132, 168, 238); NUM = (226, 170, 112)
    COMMENT = (84, 92, 120); GUT = (56, 62, 96); BI = (124, 186, 202)
    KEYWORDS = {"def", "if", "not", "in", "for", "return", "else", "and", "or", "None", "True", "False", "import", "from", "class", "while", "lambda", "zip"}
    BUILTINS = {"int", "float", "tuple", "range", "min", "max", "round", "len", "list", "self", "np", "pygame"}

    def __init__(self):
        p = os.path.join(HERE, "code_lines.json")
        self.lines = {int(k): v for k, v in json.load(open(p)).items()} if os.path.exists(p) else {}
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
        self.small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
        # (time, line being executed): the highlight walks through the render path
        self.sched = [(0.0, 229), (0.5, 230), (1.0, 230), (1.25, 231), (1.5, 232), (1.75, 234), (2.0, 235),
                      (2.5, 239), (2.75, 241), (3.75, 247), (4.25, 249), (5.25, 359), (5.5, 360), (5.75, 361),
                      (7.0, 370), (8.0, 374), (9.0, 272), (9.25, 273), (10.5, 285), (12.25, 296), (15.0, 89),
                      (15.75, 90), (16.25, 91), (16.75, 92), (17.25, 93), (17.75, 95), (18.25, 96), (18.75, 98),
                      (19.6, 104), (24.0, 104)]
        self.cache = {}

    def cur(self, t):
        ts = np.array([a for a, _ in self.sched]); ls = np.array([b for _, b in self.sched], float)
        i = np.searchsorted(ts, t, side="right") - 1
        i = int(np.clip(i, 0, len(ts) - 2))
        if abs(ls[i + 1] - ls[i]) > 30:                         # a jump to another part of the file
            k = np.clip((t - ts[i + 1] + 0.18) / 0.18, 0, 1)
            return ls[i] + (ls[i + 1] - ls[i]) * smooth(k) if t > ts[i + 1] - 0.18 else ls[i]
        k = (t - ts[i]) / max(ts[i + 1] - ts[i], 1e-6)
        return ls[i] + (ls[i + 1] - ls[i]) * k

    def colour(self, text):
        # tiny syntax colouring
        out = []
        import re
        pos = 0
        if text.strip().startswith("#"):
            return [(text, self.COMMENT)]
        parts = re.split(r"(#.*$)", text)
        for part in parts:
            if part.startswith("#"):
                out.append((part, self.COMMENT)); continue
            for tok in re.findall(r"\s+|[A-Za-z_][A-Za-z_0-9]*|\d+\.?\d*|.", part):
                if tok.strip() == "":
                    c = self.TEXT
                elif tok in self.KEYWORDS:
                    c = self.KW
                elif tok in self.BUILTINS:
                    c = self.BI
                elif re.match(r"^\d", tok):
                    c = self.NUM
                else:
                    c = self.TEXT
                out.append((tok, c))
        # function names after def
        for i in range(len(out) - 2):
            if out[i][0] == "def":
                j = i + 1
                while j < len(out) and out[j][0].strip() == "":
                    j += 1
                if j < len(out):
                    out[j] = (out[j][0], self.FN)
        return out

    def render(self, t, canvas):
        # canvas : full 720x1280 float32
        ph = OUT_H - H
        panel = Image.new("RGB", (OUT_W, ph), self.BG)
        d = ImageDraw.Draw(panel)
        d.rectangle((0, 0, OUT_W, 29), fill=self.HEAD)
        d.text((16, 6), "liminal_space_field.py", font=self.font, fill=(172, 180, 206))
        cur = self.cur(t)
        row_h, y_bar = 20, 1018 - H            # the highlight bar's centre in panel coords
        for k in range(-8, 12):
            ln = int(math.floor(cur)) + k
            y = y_bar + (ln - cur) * row_h
            if y < 30 - row_h or y > ph:
                continue
            if abs(ln - cur) < 0.5 and False:
                pass
            if ln == int(round(cur)):
                d.rectangle((0, y - 10, OUT_W, y + 10), fill=self.BAR)
                d.rectangle((0, y - 10, 3, y + 10), fill=(226, 190, 110))
            txt = self.lines.get(ln)
            d.text((14, y - 8), f"{ln:3d}", font=self.font, fill=(70, 78, 112) if ln != int(round(cur)) else (150, 160, 200))
            if txt:
                x = 43
                for tok, c in self.colour(txt.expandtabs(4)):
                    d.text((x, y - 8), tok, font=self.font, fill=c)
                    x += 8 * len(tok)
                    if x > OUT_W:
                        break
        # the header sits above the code : redraw so scrolling code goes beneath it
        d.rectangle((0, 0, OUT_W, 29), fill=self.HEAD)
        d.text((16, 6), "liminal_space_field.py", font=self.font, fill=(172, 180, 206))
        d.text((OUT_W - 16 - 8 * 16, 6), "running · 30 fps", font=self.font, fill=(222, 190, 120))
        canvas[H:] = np.asarray(panel, np.float32)


# ----------------------------------------------------------------------------- hud
def draw_hud(scene_u8, t, f, mon, outro_alpha=1.0):
    im = Image.fromarray(scene_u8)
    d = ImageDraw.Draw(im, "RGBA")
    mono = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
    tiny = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
    col = (214, 226, 232, 215)
    d.text((16, 786), "@dwcminh.ascii", font=mono, fill=col)
    d.text((16, 811), f"daisies {f.n_daisies:,}".replace(",", ""), font=mono, fill=col)
    d.text((147, 811), f"moon {int(round(mon * 100)):>3d}%", font=mono, fill=col)
    d.text((258, 811), f"fireflies {f.fireflies}", font=mono, fill=col)
    d.text((16, 831), f"letters {f.n_letters:,}", font=mono, fill=col)
    ts = f"t {t:05.2f}s"
    d.text((703 - 7 * len(ts), 12), ts, font=tiny, fill=(232, 232, 206, 235))
    if (t % 1.0) < 0.5:
        x = 703 - 7 * len(ts) - 12
        d.ellipse((x, 16, x + 5, 21), fill=(236, 204, 90, 235))
    return np.asarray(im)


# ----------------------------------------------------------------------------- watermark + outro
def overlay(img, t):
    # the creator's tag over the scene, then the whole thing dims to a title card at the end
    k = float(np.clip((t - 19.55) / 0.45, 0, 1))
    if k > 0:
        img = (img.astype(np.float32) * (1 - 0.86 * k)).astype(np.uint8)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    try:
        big = ImageFont.truetype("/usr/share/fonts/truetype/google-fonts/Anton-Regular.ttf", 21)
    except Exception:
        big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
    try:
        sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf", 14)
    except Exception:
        sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    if k < 1:
        a = int(255 * (1 - k))
        d.text((661, 376), "@DWCMINH.ASCII", font=big, fill=(255, 255, 255, a), anchor="rm")
        d.text((661, 409), "neheart, reidenshi \u2022 snowfall", font=sm, fill=(235, 235, 235, int(a * 0.8)), anchor="rm")
    if k > 0:
        d.text((360, 921), "@DWCMINH.ASCII", font=big, fill=(255, 255, 255, int(255 * k)), anchor="mm")
    return np.asarray(im)


# ----------------------------------------------------------------------------- one frame
class Renderer:
    def __init__(self):
        self.f = Field()
        self.p = Panel()

    def frame(self, i):
        t = (i % int(LOOP * FPS)) / FPS if False else i / FPS
        f = self.f
        light = moonlight(t)
        f.backdrop(t, light)
        # the trees stand against the sky
        sc = np.ascontiguousarray(f.scene.astype(np.uint8))
        tc = tuple(int(c * (0.9 + 0.5 * light)) for c in TREES)
        base = HOR + 1
        cv2.rectangle(sc, (0, HOR - 28), (W, base), tc, -1)
        for cx, hh, hw in f.trees:
            pts = np.array([(cx - hw, HOR - 24), (cx, HOR - 24 - hh), (cx + hw, HOR - 24)], np.int32)
            cv2.fillPoly(sc, [pts], tc, cv2.LINE_AA)
        f.scene = sc.astype(np.float32)
        f.plants(t, light)
        f.lights(t, light)
        # the painting, read back as letters
        small = cv2.resize(np.clip(f.scene, 0, 255), (COLS, ROWS), interpolation=cv2.INTER_AREA)
        f.letters(t, small)
        scene = np.clip(f.scene, 0, 255).astype(np.uint8)
        scene = draw_hud(scene, t, f, light)
        canvas = np.zeros((OUT_H, OUT_W, 3), np.float32)
        canvas[:H] = scene
        self.p.render(t, canvas)
        out = np.clip(canvas, 0, 255).astype(np.uint8)
        return overlay(out, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="*", help="only write these frame numbers as PNG")
    ap.add_argument("--out", default="liminal_space_field.mp4")
    ap.add_argument("--seconds", type=float, default=LOOP)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--step", type=int, default=1)
    a = ap.parse_args()
    r = Renderer()
    if a.frames:
        for n in a.frames:
            Image.fromarray(r.frame(n)).save(f"frame_{n:04d}.png")
        return
    n = int(a.seconds * FPS)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OUT_W}x{OUT_H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "14",
           "-pix_fmt", "yuv420p", a.out]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(a.start, n, a.step):
        pr.stdin.write(r.frame(i).tobytes())
        if i % 30 == 0:
            print(f"frame {i}/{n}", file=sys.stderr, flush=True)
    pr.stdin.close(); pr.wait()


if __name__ == "__main__":
    main()
