import pandas as pd
import numpy as np 
import os

# Getting the CSV into DataFrame
data = pd.read_csv('H:\capstone\Data\DataSet\oasis_cross-sectional.csv')

# Get the CDR Values
cdr_data = data.set_index("ID")[["CDR","DISC"]].to_dict(orient='index')
print(len(cdr_data))
# ALGORITHM
"""
LOOP INTO CDR 
GET THE INDEX VALUE
    BASED ON THE INDEX VALUE ACCESS THE FILE (ID)
    COMPARE THE CDR AND PASTE THE FOLLOWING INTO APPROPIATE FOLDERS
for i in range(len(cdr_data)):
    print(os.system(""))

for i in range(len(cdr_data)):
    if cdr_data[i] == 0:
        # NO DEMENTIA
        print(i)
        os.system(r"cp H:\\capstone\\Data\\DataSet\\disc1\\OAS1_"+str(i).zfill(4)+"_MR1\\PROCESSED\\MPRAGE\\T88_111\\OAS1_"+str(i).zfill(4)+"_MR1_mpr_n4_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Training-Data\\A")
    if cdr_data[i] == np.NaN:
        print("YASH IS GOD")

"""
# Preprocessing
"""
for i,j in zip(cdr_data,cdr_data.values()):
    if j["CDR"] == 0:
        print("ID = "+ str(i)+" CDR =" + str(j["CDR"]) + " DISC = " + str(j["DISC"]))
        if os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n4_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Training-Data\\no_dement") == 0:
            print("EXISTENCE")
        else:
            os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n3_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Training-Data\\no_dement") 
            print("GOD")
    elif j["CDR"] == 0.5:
        print("ID = "+ str(i)+" CDR =" + str(j["CDR"]) + " DISC = " + str(j["DISC"]))
        if os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n4_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Training-Data\\earlyonset_dementia") == 0:
            print("EXISTENCE")
        else:
            os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n3_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Training-Data\\earlyonset_dementia") 
            print("GOD")
    elif j["CDR"] == 1:
        print("ID = "+ str(i)+" CDR =" + str(j["CDR"]) + " DISC = " + str(j["DISC"]))
        if os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n4_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Training-Data\\moderate_dementia") == 0:
            print("EXISTENCE")
        else:
            os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n3_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Training-Data\\moderate_dementia") 
            print("GOD")
    elif j["CDR"] == 2:
        print("ID = "+ str(i)+" CDR =" + str(j["CDR"]) + " DISC = " + str(j["DISC"]))
        if os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n4_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Training-Data\\severe_dementia") == 0:
            print("EXISTENCE")
        else:
            os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n3_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Training-Data\\severe_dementia") 
            print("GOD")
    elif np.isnan(j["CDR"]):
        print("ID = "+ str(i)+" CDR =" + str(j["CDR"]) + " DISC = " + str(j["DISC"]))
        if os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n4_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Testing_Data") == 0:
            print("EXISTENCE")
        else:
            os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n3_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Testing_Data") 
            print("GOD")

        print("pasted")
"""
# LOCKS

hc_data_lock = 0
mil_data_lock = 0
mod_data_lock = 0
sev_data_lock = 0


# Training Data
for i,j in zip(cdr_data,cdr_data.values()):
    if j["CDR"] == 0:
        if hc_data_lock < 68:
            for k in range(0,10):
                print("ID = "+ str(i)+" CDR =" + str(j["CDR"]) + " DISC = " + str(j["DISC"])+ " GODLIKE = " +str(k))
                if os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n"+str(k)+"_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Training-Data\\HC\\") == 0:
                    print("EXISTENCE OF KING")
                    hc_data_lock = hc_data_lock + 1
                    break
        else:
            for k in range(0,10):
                print("ID = "+ str(i)+" CDR =" + str(j["CDR"]) + " DISC = " + str(j["DISC"])+ " GODLIKE = " +str(k))
                if os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n"+str(k)+"_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Testing_Data\\HC\\") == 0:
                    print("EXISTENCE OF KING")
                    hc_data_lock = hc_data_lock + 1
                    break

            """
            else:
                os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n3_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Training-Data\\HC\\") 
                print("GOD")
            """
    elif j["CDR"] > 0:
        if j["CDR"] == 0.5:
            if mil_data_lock < 35:
                for k in range(0,10):
                    print("ID = "+ str(i)+" CDR =" + str(j["CDR"]) + " DISC = " + str(j["DISC"]))
                    if os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n"+str(k)+"_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Training-Data\\AD\\") == 0:
                        print("EXISTENCE")
                        mil_data_lock = mil_data_lock + 1
                        break
            else:
                for k in range(0,10):
                    print("ID = "+ str(i)+" CDR =" + str(j["CDR"]) + " DISC = " + str(j["DISC"]))
                    if os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n"+str(k)+"_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Testing_Data\\AD\\") == 0:
                        print("EXISTENCE")
                        mil_data_lock = mil_data_lock + 1
                        break
        if j["CDR"] == 1:
            if mod_data_lock < 14:
                for k in range(0,10):
                    print("ID = "+ str(i)+" CDR =" + str(j["CDR"]) + " DISC = " + str(j["DISC"]))
                    if os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n"+str(k)+"_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Training-Data\\AD\\") == 0:
                        print("EXISTENCE")
                        mod_data_lock = mod_data_lock + 1
                        break
            else:
                for k in range(0,10):
                    print("ID = "+ str(i)+" CDR =" + str(j["CDR"]) + " DISC = " + str(j["DISC"]))
                    if os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n"+str(k)+"_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Testing_Data\\AD\\") == 0:
                        print("EXISTENCE")
                        mod_data_lock = mod_data_lock + 1
                        break
        if j["CDR"] == 2:
            if sev_data_lock < 1:
                for k in range(0,10):
                    print("ID = "+ str(i)+" CDR =" + str(j["CDR"]) + " DISC = " + str(j["DISC"]))
                    if os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n"+str(k)+"_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Training-Data\\AD\\") == 0:
                        print("EXISTENCE")
                        sev_data_lock = sev_data_lock + 1
                        break
            else:
                for k in range(0,10):
                    print("ID = "+ str(i)+" CDR =" + str(j["CDR"]) + " DISC = " + str(j["DISC"]))
                    if os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n"+str(k)+"_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Testing_Data\\AD\\") == 0:
                        print("EXISTENCE")
                        sev_data_lock = sev_data_lock + 1
                        break
    elif np.isnan(j["CDR"]):
        for k in range(0,10):
            print("ID = "+ str(i)+" CDR =" + str(j["CDR"]) + " DISC = " + str(j["DISC"]))
            if os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+str(j["DISC"])+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n"+str(k)+"_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Unseen_data\\") == 0:
                print("EXISTENCE")
                break
            else:
                os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+str(j["DISC"])+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n"+str(k)+"_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Unseen_data\\") 
                print("GOD")
                break


    print(mil_data_lock, mod_data_lock, sev_data_lock, hc_data_lock)
    """
    elif j["CDR"] == 1:
        print("ID = "+ str(i)+" CDR =" + str(j["CDR"]) + " DISC = " + str(j["DISC"]))
        if os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n4_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Training-Data\\moderate_dementia") == 0:
            print("EXISTENCE")
        else:
            os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n3_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Training-Data\\moderate_dementia") 
            print("GOD")
    elif j["CDR"] == 2:
        print("ID = "+ str(i)+" CDR =" + str(j["CDR"]) + " DISC = " + str(j["DISC"]))
        if os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n4_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Training-Data\\severe_dementia") == 0:
            print("EXISTENCE")
        else:
            os.system(r"cp -R H:\\capstone\\Data\\DataSet\\"+j["DISC"]+"\\"+ str(i) +"\\PROCESSED\\MPRAGE\\T88_111\\"+str(i)+"_mpr_n3_anon_111_t88_gfc_sag_95.gif H:\\capstone\\Training-Data\\severe_dementia") 
            print("GOD")
    """


