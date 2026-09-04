import numpy as np

while True:
    s = input("values: ")
    x = np.array([float(v) for v in s.replace(",", " ").split()])
    # print(f"n = {len(x)}")
    # print(f"mean = {x.mean():.6g}")
    # print(f"std  = {x.std(ddof=1):.6g}   (sample, ddof=1)")
    # print(f"std  = {x.std(ddof=0):.6g}   (population, ddof=0)")
    mean = x.mean()
    std = x.std(ddof=1)
    print(f"{mean:.4f} +/- {std:.4f}")