from avera.datamanagement.SymbolDataManager import SymbolDataManager
from avera.envs.SimpleEnv import SimpleEnv

df = SymbolDataManager().getData("TEST", "H1")
#featureScaler = FeatureScaler()
#df = featureScaler.extractFeature(df)

simpleEnv = SimpleEnv(df)
obs = simpleEnv.reset()
print( str(obs[0]) + " | " + str(obs[1]) )

#check buy
obs, reward, done, info = simpleEnv.step(0)
print( str(obs[0]) + " | " + str(obs[1]) )
print( reward )

#check hold
obs, reward, done, info = simpleEnv.step(1)
print( str(obs[0]) + " | " + str(obs[1]) )
print( reward )

#check sell
obs, reward, done, info = simpleEnv.step(2)
print( str(obs[0]) + " | " + str(obs[1]) )
print( reward )

