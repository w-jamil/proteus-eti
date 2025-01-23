import pandas as pd
import cilm as lm



####################################################################################################

# fred = Fred(api_key='5e4a18b3ce1c062376bf966bea553db9')
# data = fred.get_series('FEDFUNDS')
fred = pd.read_csv("C:\\Users\\lenovo\\OneDrive\\Documents\\academia\\extremexp\\cilm\\data\\us_rates.csv")
fred.columns = ['DATE',"RATES"]
fred['DATE'] = pd.to_datetime(fred['DATE'],dayfirst=True)
fred.index = fred.DATE
data = fred.drop(['DATE'],axis=1)
data = pd.Series(data['RATES'])
cilm1 = lm.CILM(20,data,"RATES","DATE",0.95)
out = cilm1.sim()
out.index = pd.to_datetime(out["DATE"])
out = out.drop(["DATE"],axis=1)
classical_case = out[["ACTUAL","NLCI","NUCI"]]
######################################################################################
# If Y is outside the interval assign -1 else 1
# Then we count the 1's and -1's to see the amount of times the forcast was outsie and inside the interval
####################################################################################
cc = []

for i in range(len(classical_case)):
  if (classical_case["ACTUAL"].iloc[i] < classical_case["NLCI"].iloc[i]) or (classical_case["ACTUAL"].iloc[i] > classical_case["NUCI"].iloc[i]):
    cc.append(-1)
  else:
    cc.append(1)
machine_case = out[["ACTUAL","CLCI","CUCI"]]
###############################################################################
# If Y is outside the interval assign -1 else 1
# Then we count the 1's and -1's to see the amount of times the forcast was outsie and inside the interval
###############################################################################
mc = []
for i in range(len(machine_case)):

  if (machine_case["ACTUAL"].iloc[i] < machine_case["CLCI"].iloc[i]) or (machine_case["ACTUAL"].iloc[i] > machine_case["CUCI"].iloc[i]):
    
    mc.append(-1)

  else:

    mc.append(1)
output = pd.concat([pd.Series(mc),pd.Series(cc)],axis=1)
output.columns = ["MACHINE","CLASSICAL"]

print("PERCENTAGE OF CILM OUTSIDE:")
print(round(abs(output[output.MACHINE == -1]["MACHINE"].sum())/len(output),4))


print("ACCURACY OF CILM:")
print(round(1-abs(output[output.MACHINE == -1]["MACHINE"].sum())/len(output),4))


print("CALIBRATION ERROR CILM:")
print(round(abs(0.95-abs(output[output.MACHINE == -1]["MACHINE"].sum())/len(output),4)))


print("PERCENTAGE OF NCI OUTSIDE:")
print(round(abs(output[output.CLASSICAL == -1]["CLASSICAL"].sum())/len(output),4))

print("ACCURACY OF NCI:")
print(round(1-abs(output[output.CLASSICAL == -1]["CLASSICAL"].sum())/len(output),4))

print("CALIBRATION ERROR NCI:")
print(round(abs(0.95-abs(output[output.CLASSICAL == -1]["CLASSICAL"].sum())/len(output),4)))

############################################################

