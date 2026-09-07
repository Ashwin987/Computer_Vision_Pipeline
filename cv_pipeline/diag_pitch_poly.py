orig     = [(110,1035),(265,275),(910,260),(1640,915)]
expanded = [(110,1035),(265,275),(1750,260),(1900,1035)]

players = {2:(1172,482),3:(1290,489),4:(524,599),5:(1426,558),6:(1004,736),
           7:(1281,592),8:(536,442),9:(774,509),10:(1534,489),11:(802,541),
           12:(1551,599),13:(838,380),14:(366,566),15:(1678,610),
           18:(1447,535),19:(1021,495),20:(1063,499),21:(1298,512),38:(722,408)}

def sh_inside(p, a, b):
    return ((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])) >= 0

def inside_poly(pt, poly):
    n = len(poly)
    return all(sh_inside(pt, poly[i], poly[(i+1)%n]) for i in range(n))

print("=== Original polygon ===")
for pid, pos in sorted(players.items()):
    tag = "IN" if inside_poly(pos, orig) else "OUT"
    print(f"  PID {pid:>3}  {str(pos):>14}  {tag}")

print()
print("=== Expanded polygon [(110,1035),(265,275),(1750,260),(1900,1035)] ===")
for pid, pos in sorted(players.items()):
    tag = "IN" if inside_poly(pos, expanded) else "OUT"
    print(f"  PID {pid:>3}  {str(pos):>14}  {tag}")

print()
print("=== Convexity check (expanded) — all cross products must be > 0 ===")
n = len(expanded)
for i in range(n):
    a = expanded[i-1]; b = expanded[i]; c = expanded[(i+1)%n]
    cross = (b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0])
    print(f"  vertex {i} {str(b):>15}  cross={cross:>10}  {'CCW OK' if cross>0 else 'PROBLEM'}")
