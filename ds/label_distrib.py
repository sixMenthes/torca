import polars as pl
import matplotlib.pyplot as plt


df = pl.read_csv("./DCLDE_w_Buzzes.csv", null_values=["NA"], ignore_errors=True)

