import pyagrum as gum
import pyagrum.lib.notebook as gnb
import pandas as pd

bn = gum.randomBN(n=20, domain_size=4)
generator = gum.BNDatabaseGenerator(bn)
n = 1000
generator.drawSamples(n)
df = generator.to_pandas()
print(df)