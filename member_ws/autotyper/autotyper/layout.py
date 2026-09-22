# layout.py -- the Redragon K552 key layout in key units (INTERFACES.md section 6.3)
# public keyboard-standard information, not a secret of the rig. 1u = 19.05mm.
# a key's cell has its top-left at (x, y) and width w, height 1.
U = 0.01905          # metres per key unit
GRID_W, GRID_H = 18.25, 6.25   # whole key grid, in units


def row(y, items):
    #turn a row's (name, x, width) triples into (name, x, y, width) at row y
    result = []
    for n, x, w in items:
        result.append((n, x, y, w))
    return result


def build():
    #lay out every key
    k = []
    k += row(0, [
        ('ESC',       0,     1),
        ('F1',        2,     1),
        ('F2',        3,     1),
        ('F3',        4,     1),
        ('F4',        5,     1),
        ('F5',        6.5,   1),
        ('F6',        7.5,   1),
        ('F7',        8.5,   1),
        ('F8',        9.5,   1),
        ('F9',        11,    1),
        ('F10',       12,    1),
        ('F11',       13,    1),
        ('F12',       14,    1),
        ('PRTSC',     15.25, 1),
        ('SCRLK',     16.25, 1),
        ('PAUSE',     17.25, 1),
    ])
    k += row(1.25, [
        ('`',         0,     1),
        ('1',         1,     1),
        ('2',         2,     1),
        ('3',         3,     1),
        ('4',         4,     1),
        ('5',         5,     1),
        ('6',         6,     1),
        ('7',         7,     1),
        ('8',         8,     1),
        ('9',         9,     1),
        ('0',         10,    1),
        ('-',         11,    1),
        ('=',         12,    1),
        ('BACKSPACE', 13,    2),
        ('INS',       15.25, 1),
        ('HOME',      16.25, 1),
        ('PGUP',      17.25, 1),
    ])
    k += row(2.25, [
        ('TAB',       0,     1.5),
        ('Q',         1.5,   1),
        ('W',         2.5,   1),
        ('E',         3.5,   1),
        ('R',         4.5,   1),
        ('T',         5.5,   1),
        ('Y',         6.5,   1),
        ('U',         7.5,   1),
        ('I',         8.5,   1),
        ('O',         9.5,   1),
        ('P',         10.5,  1),
        ('[',         11.5,  1),
        (']',         12.5,  1),
        ('\\',        13.5,  1.5),
        ('DEL',       15.25, 1),
        ('END',       16.25, 1),
        ('PGDN',      17.25, 1),
    ])
    k += row(3.25, [
        ('CAPS',      0,     1.75),
        ('A',         1.75,  1),
        ('S',         2.75,  1),
        ('D',         3.75,  1),
        ('F',         4.75,  1),
        ('G',         5.75,  1),
        ('H',         6.75,  1),
        ('J',         7.75,  1),
        ('K',         8.75,  1),
        ('L',         9.75,  1),
        (';',         10.75, 1),
        ("'",         11.75, 1),
        ('ENTER',     12.75, 2.25),
    ])
    k += row(4.25, [
        ('LSHIFT',    0,     2.25),
        ('Z',         2.25,  1),
        ('X',         3.25,  1),
        ('C',         4.25,  1),
        ('V',         5.25,  1),
        ('B',         6.25,  1),
        ('N',         7.25,  1),
        ('M',         8.25,  1),
        (',',         9.25,  1),
        ('.',         10.25, 1),
        ('/',         11.25, 1),
        ('RSHIFT',    12.25, 2.75),
        ('UP',        16.25, 1),
    ])
    k += row(5.25, [
        ('LCTRL',     0,     1.25),
        ('LWIN',      1.25,  1.25),
        ('LALT',      2.5,   1.25),
        ('SPACE',     3.75,  6.25),
        ('RALT',      10,    1.25),
        ('RWIN',      11.25, 1.25),
        ('MENU',      12.5,  1.25),
        ('RCTRL',     13.75, 1.25),
        ('LEFT',      15.25, 1),
        ('DOWN',      16.25, 1),
        ('RIGHT',     17.25, 1),
    ])
    return k


KEYS = build()                                  # [(name, x, y, w), ...] in units

BY_NAME = {}                                     # fast lookup by key name
for n, x, y, w in KEYS:
    BY_NAME[n] = (x, y, w)

CHAR_KEYS = {}                                    # maps a launch-key character to its key name
for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
    CHAR_KEYS[c] = c
CHAR_KEYS[' '] = 'SPACE'


def key_center_units(name):
    #centre of a key's cell, in units from the grid's top-left: (x + w/2, y + 0.5)
    x, y, w = BY_NAME[name]
    return x + w / 2.0, y + 0.5


def key_at_units(gx, gy, inset=0.0):
    #name of the key whose cell contains grid point (gx, gy) units
    for n, x, y, w in KEYS:
        # inset shrinks the cell so near-misses near a border don't count as a hit
        x_ok = x + inset <= gx and gx <= x + w - inset
        y_ok = y + inset <= gy and gy <= y + 1 - inset
        if x_ok and y_ok:
            return n
    return None