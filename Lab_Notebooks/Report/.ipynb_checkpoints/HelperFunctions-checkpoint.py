from datetime import datetime
import pandas as pd
pd.options.mode.chained_assignment = None # avoid error 
import numpy as np 
### Mendota data ###
# Parameters 

maxThermocline= 14  
years = range(1995,2017) 

#############################################
#                                           #
######### FIRST  Helper function ############
#                                           #
#############################################


def DataPreparation_Mendota(StrataSource):
    if StrataSource == 'CD': #LTER
        # Preprocessing
        p = 'Input'
        df = pd.read_csv(p+'/ntl29_v5.csv', low_memory = False)
        """Convert Datetime objects to seconds for numerical/quantitative parsing"""
        df['sampledate'] = pd.to_datetime(df['sampledate']) #
        ''' Filtering Lake Mendota '''
        df = df[df['lakeid'].isin(['ME'])] # keep lakeid values equal to ME
        """ Select columns of interet """
        df = df.iloc[:,np.r_[3:5,8:14]] # Select only columns to be used
        ''' Drop na'''
        df.dropna(thresh=3, axis=0, inplace = True) # Drop Na when na.sum()>2  ?? REVIEW, the result are okey but I do not get the code :c ??
        ''' Resteindex'''
        df.reset_index(drop=True,inplace = True) # Restart the index after eliminate rows, hence from 0 to naa
        ''' Generate data sequence '''
        dataseq = pd.DataFrame({'sampledate':pd.date_range(start='1995-05-09',end='2016-11-09')}).set_index('sampledate')
        
        ## Epilimnetic Temoperature calculation
        ''' Mean temperature at 0 depth, condidering the Na values'''
        ET= (df[df.depth==0].groupby('sampledate').agg({'wtemp':lambda x: x.mean(skipna=False)})).rename(columns = {'wtemp':'EpiTemp'})
        # Merge EpiTemp and dataseq which will generate Na's
        EpiTemp =pd.merge(ET,dataseq,how = 'right',left_index=True, right_index=True)
        # Interpolate those Na's
        EpiTemp.interpolate(method='linear', axis=0, inplace=True)
    
        ## Hypolimnetic Temoperature calculation
        
        HT= df[df.depth>=20].groupby('sampledate').agg({'wtemp':lambda x: x.mean(skipna=False)}).rename(columns = {'wtemp':'HypoTemp'})
        HypoTemp =pd.merge(HT,dataseq,how = 'right',left_index=True, right_index=True)
        HypoTemp.interpolate(method='linear', axis=0, inplace=True)
    
        # Merging Epi and Hypo temperatue in same data frame
        df = pd.merge(EpiTemp,HypoTemp,how = 'left',left_index=True, right_index=True)
    
        ## Stratification column
        df['Strat'] =  ((df.EpiTemp - df.HypoTemp)>=2)
        df['Therm'] = 1  # Initialize Thermocline to then calculate it 
        
        for y in years:
            stratYear = df[df.index.year == y]
            n = len(stratYear[stratYear.Strat== True])
    
        
            test = stratYear[stratYear.Strat == True].HypoTemp
            zi = (test-min(test))/(max(test)-min(test)) * maxThermocline + 1 #weird way of scaling thermocline shape 
            stratYear.Therm[stratYear.Strat== True] = zi
  
            df[df.index.year == y] = stratYear # replace with new values }
      
    else: 
         # Using the data from GLMdata
        p = 'Input'
        GLMdata = pd.read_csv(p +'/Modeled_thermocline_daily.csv')
        dataseq = pd.DataFrame({'sampledate':pd.date_range(start='1995-05-09',end='2015-12-31')}).set_index('sampledate')
        GLMdata.rename(columns = {'datetime':'sampledate', 'thermo.depth':'Therm'}, inplace = True)
        df = GLMdata.copy()
        df['Strat'] = True
        df['Therm'].fillna(1, inplace = True)
        df.loc[df['Therm'] == 1, 'Strat'] = False 
        df = df[['sampledate','EpiTemp','HypoTemp','Strat','Therm']]
        df['sampledate'] = pd.to_datetime(df['sampledate'])
        df.set_index('sampledate', inplace=True)
    return(df,dataseq)



#############################################
#                                           #
######### Second Helper function ############
#                                           #
#############################################




def GetInFlow(dataseq,HydroSource):  # From Yahara river 
    LakeArea = 39393719 # For normalizing plots
    p = 'Input/'
    dataseq = pd.DataFrame({'sampledate':pd.date_range(start='1995-05-09',end='2016-11-09')}).set_index('sampledate')
    # PLotIt = True
    if HydroSource==1:
        Pfactor = 1.4 #USGS
    else: Pfactor = 1.0
    
    if HydroSource== 1: 
        # Preprocessing
        yahara = pd.read_csv(p+'/USGS-05427718_NWIS_Yahara_Windsor.csv', low_memory = False)
        dataseq = pd.DataFrame({'sampledate':pd.date_range(start='1995-05-09',end='2016-11-09')}).set_index('sampledate') # drop this line
        yahara = yahara.iloc[:,np.r_[0,3,5]]
        yahara['datetime'] = pd.to_datetime(yahara['datetime']) 
        yahara.set_index('datetime',inplace =True)
        df = pd.merge(yahara,dataseq,how = 'right',left_index=True, right_index=True)
        df['Discharge.m3.d'] = df['Discharge_ft3_s']*5*(0.0283168*3600*24)
        df['P.g.day'] = Pfactor*df['Phosphorus_unfiltered_mg_L']*df['Discharge.m3.d']
        df['Discharge.m3.d'].interpolate(method='linear', axis=0, inplace=True)
        df['P.g.day'].interpolate(method='linear', axis=0, inplace=True)
        df = df.iloc[:,-2:]
    else: 
        #
        yahara = pd.read_csv(p+'Mendota_yahara_30year2xp.csv', low_memory = False)
        yahara = yahara.iloc[:,np.r_[0,1,4:8]]

        yahara['time'] = pd.to_datetime(yahara['time']) 
        yahara.set_index('time',inplace =True)
        df = pd.merge(yahara,dataseq,how = 'right',left_index=True, right_index=True)
        df['Discharge.m3.d'] = df['FLOW']*60*1440
        df['P.g.day'] = Pfactor*(df.iloc[:, 1:5].sum(axis=1))*df['Discharge.m3.d']*31/1000
        df['Discharge.m3.d'].interpolate(method='linear', axis=0, inplace=True)
        df['P.g.day'].interpolate(method='linear', axis=0, inplace=True)
        df = df.iloc[:,5:7]

        yaharaPIHM = pd.read_csv(p+'Mendota_pheasant_branch_30year2xp.csv', low_memory = False)
        yaharaPIHM = yaharaPIHM.iloc[:,np.r_[0,1,4:8]]
        yaharaPIHM
        yaharaPIHM.rename(columns = {'time':'sampledate'},inplace= True)
        PIHM = yaharaPIHM.copy()
        PIHM['sampledate'] = pd.to_datetime(PIHM['sampledate'])
        PIHM.set_index('sampledate',inplace =True)
        PIHM = pd.merge(PIHM,dataseq,how = 'right',left_index=True, right_index=True)

        PIHM['Discharge.m3.d_2'] = PIHM['FLOW']*60*1440
        PIHM['P.g.day_2'] = Pfactor*(PIHM.iloc[:, 1:5].sum(axis=1))*PIHM['Discharge.m3.d_2']*31/1000 # g/day
        PIHM['Discharge.m3.d_2'].interpolate(method='linear', axis=0, inplace=True)
        PIHM['P.g.day_2'].interpolate(method='linear', axis=0, inplace=True)
        PIHM= PIHM.iloc[:,5:7]

        PIHM_tot = pd.merge(df,PIHM,how = 'right',left_index=True, right_index=True)
        PIHM_tot['total_load'] = PIHM_tot['P.g.day_2'] + PIHM_tot['P.g.day']
        PIHM_tot['total_discharge'] = PIHM_tot['Discharge.m3.d_2'] + PIHM_tot['Discharge.m3.d']
        df = PIHM_tot.iloc[:,-2:] #?
        df.rename(columns = {'total_load':'P.g.day','total_discharge':'Discharge.m3.d'}, inplace = True)
        df
    return(df)

#############################################
#                                           #
######### Third  Helper function ############
#                                           #
#############################################

def GetLakeData(HypoSource):
    p = 'Input/'
    obsCD = pd.read_csv(p+'chemphys.csv', low_memory = False)

    minDepthEpi = 1
    maxDepthEpi = 4
    minDepthHypo = 13
    maxDepthHypo = 18    

    Epi_mask = (obsCD.depth >= minDepthEpi) & (obsCD.depth <= maxDepthEpi ) &(~obsCD['totpuf_sloh'].isna())
    obsCDEpi= obsCD[Epi_mask][['sampledate','totpuf_sloh']]
    obsCDEpi['sampledate'] = pd.to_datetime(obsCDEpi['sampledate'])#.dt.strftime('%d-%m-%Y %H:%M:%S') ###
    obsCDEpi['totpuf_sloh'] = obsCDEpi.groupby(obsCDEpi.sampledate)['totpuf_sloh'].transform('mean')
    obsCDEpi.drop_duplicates(["sampledate"],inplace = True)
    obsCDEpi.reset_index(inplace=True)
    #obsCDEpi.set_index('sampledate', inplace = True)
    obsCDEpi.drop('index',axis = 1, inplace =True) ###

    Hypo_mask = (obsCD.depth >= minDepthHypo) & (obsCD.depth <= maxDepthHypo ) &(~obsCD['totpuf_sloh'].isna())
    obsCDHypo= obsCD[Hypo_mask][['sampledate','totpuf_sloh']]
    obsCDHypo['sampledate'] = pd.to_datetime(obsCDHypo['sampledate'])#.dt.strftime('%d-%m-%Y %H:%M:%S') ###
    obsCDHypo['totpuf_sloh'] = obsCDHypo.groupby(obsCDHypo.sampledate)['totpuf_sloh'].transform('mean')
    obsCDHypo.drop_duplicates(["sampledate"],inplace = True)
    obsCDHypo.reset_index(inplace=True)
    obsCDHypo.drop('index',axis = 1, inplace =True) ####
    #obsCDHypo.set_index('sampledate', inplace = True)
    
    Peaks = [13,31,48,66,82,99,115,130,145,161,175,191,208,223,237,253,266,283,295,307,322]
    Peaks = [x-1 for x in Peaks]

    PeakDates = obsCDEpi.loc[Peaks].sampledate
    PeakEpi = obsCDEpi[obsCDEpi.sampledate.isin(PeakDates)].totpuf_sloh
    PeakHypo = obsCDEpi[obsCDEpi.sampledate.isin(PeakDates)].totpuf_sloh
    
    if HypoSource == 1:
        HypsoNTL = pd.read_csv(p+'ntl301_hypso.csv', low_memory = False)
        HypsoNTL = HypsoNTL[HypsoNTL['lakeid'].isin(['ME'])]
        HypsoNTL.reset_index(inplace = True)

        HypsoNTL['Hypso_cum']=HypsoNTL['hp_factor'].cumsum()
        HypsoNTL['depth'] = round(HypsoNTL['depth'], 1)
        HypsoNTL['hypso'] = HypsoNTL['hp_factor']
        HypsoME = HypsoNTL[['depth','Hypso_cum','hypso']]

        # Adding 0 values for depth, hypso_cum and hypso 
        HypsoME.set_index('depth', inplace = True)
        HypsoME.loc[0] = [0,0]
        HypsoME.sort_index(inplace=True)
        HypsoME.reset_index(inplace = True)
        depth = pd.DataFrame(np.linspace(start=0, stop=25, num=251)).rename(columns = {0:'depth'})
        NewHypso = pd.merge(depth,HypsoME,how = 'outer',on = 'depth')
        NewHypso['Hypso_cum'].interpolate(method='linear', axis=0, inplace=True)
        NewHypso.drop(['hypso'], axis=1, inplace = True)
    else:
        HypsoME = None
        NewHypso = None
            
    return(HypsoME,NewHypso,obsCDEpi,obsCDHypo,PeakDates,PeakEpi,PeakHypo)

#############################################
#                                           #
######### *  Helper function     ############
#           Model configurat                #
#############################################

def ModelConfig(ModelConfig):
    '''output: LakeConfig, pars,HydroSource,StratSource,HypsoSource=''';
    '''It is runned just before Model since it ''';
    LakeConfig = pd.DataFrame([0], columns =['i'])

    #### Lake Values ####
    LakeConfig['LoadAsAdsorbed'] = 0.5 # 0.4, Proportion of load that is adsorbed
    LakeConfig['LakeArea'] = 39393719 #m2
    LakeConfig['MeanDepth'] = 12.8 #m
    LakeConfig['zm'] = 25.3 #max depth in m
    LakeConfig['SedimentArea'] = 39393719 #m2
    LakeConfig['SedimentDepth'] = 0.1 #m Per conversation with Dick Lathrop
    LakeConfig['density'] = 1.33*(100**3) #g/m^3 googled standard value for silt/clay
    
    if ModelConfig == 1:
        #        
        HydroSource = 1 #1=USGS, 2=PIHM   -> GetINflow F(x2)
        StratSource = 2 #CD=LTER, 2=GLM    -> GetStratification F(x1)
        HypoSource = 1 #1=LTER, anything else, model calculates -> GetLakeData F(x3)
        LakeConfig['Burial'] = 8.00 # for USGS loads AND P density of 0.08 mg/g
        LakeConfig['AvailableP'] = 0.08 #1.0 # Per Dick Lathrop; 0.08 per Aviah, 0.06 # mg of P/ g of dry weight
        pars = (0.01843370, 0.29054440, 1.03469700, 1.1763891) # Fit epi only full time series
    
    elif ModelConfig == 2:
        #        
        HydroSource = 2 #1=USGS, 2=PIHM   -> GetINflow F(x2)
        StratSource = 2 #CD=LTER, CD=GLM    -> GetStratification F(x1)
        HypoSource = 1 #1=LTER, anything else, model calculates -> GetLakeData F(x3)
        LakeConfig['Burial'] = 1.2 # for PIHM loads AND P density of 0.08 mg/g
        LakeConfig['AvailableP'] = 0.806 #1.0 # Per Dick Lathrop; 0.08 per Aviah, 0.06 # mg of P/ g of dry weight
        pars = (0.01843370, 0.29054440, 1.03469700, 1.1763891) # Fit epi only full time series
        
    elif ModelConfig == 3:
        #
        HydroSource = 1 #1=USGS, 2=PIHM   -> GetINflow F(x2)
        StratSource = 2 #CD=LTER, 2=GLM    -> GetStratification F(x1)
        HypoSource = 1 #1=LTER, anything else, model calculates -> GetLakeData F(x3)
        LakeConfig['Burial'] = 0.60 # for USGS loads AND P density of 0.08 mg/g
        LakeConfig['AvailableP'] =1.0 #1.0 # Per Dick Lathrop; 0.08 per Aviah, 0.06 # mg of P/ g of dry weight
        pars = (0.01874282, 0.02181110, 1.03229375, 1.15455945) # Fit epi only full time series
    
    elif ModelConfig == 4:
        #
        HydroSource = 2 #1=USGS, 2=PIHM   -> GetINflow F(x2)
        StratSource = 2 #CD=LTER, 2=GLM    -> GetStratification F(x1)
        HypoSource = 1 #1=LTER, anything else, model calculates -> GetLakeData F(x3)
        LakeConfig['Burial'] = 0.80 # for USGS loads AND P density of 0.08 mg/g
        LakeConfig['AvailableP'] =1.0 #1.0 # Per Dick Lathrop; 0.08 per Aviah, 0.06 # mg of P/ g of dry weight
        pars = (0.03835725, 0.03893900, 1.01429434, 1.11430804) # Fit epi only full time series
        
    return(LakeConfig, pars,HydroSource,StratSource,HypoSource)


#############################################
#                                           #
######### forth  Helper function ############
#           Model configurat                #
#############################################


def SedimentEquilibrium (myLake,myParameters,TAdjEpi,TAdjHypo):
    #
    LakeArea = myLake.Area
    MeanDepth = myLake.Zmean
    RT = myLake.RT
    Pepi = myLake.P
    PRetention = myParameters.PRetention
    Csed = myParameters.TPcSed
    Crecycle = myParameters.TPcRecycle
    SoilDeposition = myParameters.SoilDeposition
    SedDepth = myParameters.TPSedDepth
    SedAvailP = myParameters.TPSedAvailP
    BulkDensity = myParameters.TPSedBulkDensity
    PEpi_g = Pepi*LakeArea*MeanDepth # g
    ShowMendotaExample = False
    
    if ShowMendotaExample == False:
        #
        LakeP = 0.14 # g/m3
        LakeArea = 3.961e7 # m2
        MeanDepth = 12.8 # m
        RT = 4 # y
        PRetention = 0.73 # retention of P load
        SedDepth = 0.1 # m
        SoilDeposition = 1.37 # mm/y
        TAdjEpi = 0.54 # Unitless
        TAdjHypo = 0.4 # Unitless
        Csed = 0.015 # 1/d
        Crecycle = 0.00008 # 1/d
        # Approach 1, based on water column P and assumed sediment, recycling
        PsedEquil = (PEpi_g * Csed * 365 * TAdjEpi) /(Crecycle * 365 * TAdjHypo + (SoilDeposition/1000) * 1/SedDepth)
        PsedEquilAreal = PsedEquil / LakeArea
        # Approach 2, based on Vollenweider estimate of load
        PLoadVoll = Pepi * (MeanDepth*(1/RT + PRetention)) # g/m2/y
        PsedEquilVoll = PLoadVoll*PRetention / ((SoilDeposition/1000) * 1/SedDepth)
        # Approach 3, based on empirical value for P content of sediment 'soil'

        '''SedArealAssumed = myParameters$TPSedAvailP * (1/1000) * myParameters$TPSedBulkDensity  * myParameters$TPSedDepth''';
    Psed = (PEpi_g * Csed * TAdjEpi * 365) / (Crecycle * TAdjHypo * 365 + (SoilDeposition/1000) * 1/SedDepth)
    PsedAreal = Psed/LakeArea
    # Approach 2, substituting Vollenweider load and assumed retention into Equation 2 above
    PLoadVoll = Pepi * (MeanDepth*(1/RT + PRetention)) # g/m2/y
    PsedVollAreal = PLoadVoll*PRetention / ((SoilDeposition/1000) * 1/SedDepth)
    # Approach 3, based on empirical value for P content of sediment 'soil'
    PsedEmpiricalAreal = SedAvailP * (1/1000) * BulkDensity  * SedDepth
    
    return(PsedEmpiricalAreal,PsedAreal,PsedVollAreal)


#############################################
#                                           #
#########   MechanisticModels    ############
#                                           #
#############################################

def MechanisticModels (LakeConfig,pars,HydroSource,StratSource,HypoSource,StratData,Inflow,HypsoME,NewHypso,obsCDEpi,obsCDHypo,PeakDates,PeakEpi,PeakHypo ):
    #
    #### Possible input parameters ####
    obsTherm = pd.DataFrame(round(StratData['Therm'], 1))
    Csed = pars[0] #Sedimentation Rate ((12.5)*(1/1000)*(1/.3) # pats dissertation)
    b = pars[1] #second constant in Nurnberg Recycling Eq
    a = -4.3 #pars [3] #original is -4.3, first constant in Nurnberg Recycling Eq
    ThetaS = pars[2] # Theta for sedimentation
    #ThetaS = 1
    TbaseS = 10 #26 #pars[4]
    ThetaR = pars[3] # Theta for recycling (~1)
    TbaseR = 10 #20 #pars[6]
    maxThermocline = 14 # Maximum depth of thermocline - 1
    EpiFactor = 0.0 # Recycling into the epilimnion? 

    Burial = (LakeConfig.Burial).item() # for PIHM loads AND P density of 0.08 mg/g
    AvailableP = (LakeConfig.AvailableP).item() # Per Dick Lathrop; 0.08 per Aviah, 0.06 # mg of P/ g of dry weight

    ### Ndays ###
    #obsCDEpi.set_index('sampledate',inplace = True)
    #(obsCDEpi.index[-1] - obsCDEpi.index[0]).days
    Tdays = int(str(StratData.reset_index()['sampledate'].iloc[-1]-StratData.reset_index()['sampledate'].iloc[0]).split(' ')[0])
    Ndays= range(0,Tdays+1)

    #### Lake Config Values ####
    LakeArea = (LakeConfig.LakeArea).item() #m2
    MeanDepth = (LakeConfig.MeanDepth).item() #m
    # nex line is to generate a variable with the same value (25) but with a len of obsTherm to not get error in calculation
    zm = (LakeConfig.zm).item()#max depth in m

    #### Sediment values ####
    LoadAsAdsorbed = (LakeConfig.LoadAsAdsorbed).item() # Proportion of load that is adsorbed
    SedimentArea = (LakeConfig.SedimentArea).item()  #m2
    SedDepth = (LakeConfig.SedimentDepth).item()  #m Per conversation with Dick Lathrop
    #Dry density of Lake Sediment
    density = (LakeConfig.density).item() #g/m^3 googled standard value for silt/clay

    LakeVolume = (LakeArea* MeanDepth) #m3
    SedV = SedimentArea * SedDepth #m3
    SedimentPConc = AvailableP * density * (1/1000)*SedDepth # g/m2 of sediment 
    
     #####
        
     #If hypso not passed to function, then use following from the literature
    if HypsoME.Hypso_cum.values is None:
        
        # [0] -> to just grab the first value of zm and not get and error for doing math with variables with diff shape
        A = ((3*((2*(MeanDepth/zm))-1))*(((zm-obsTherm.Therm)/zm)**2)) - ((2*((3*(MeanDepth/zm))-2))*((zm-obsTherm.Therm)/zm))
        V = (((zm-obsTherm.Therm)/zm)+(((3*(MeanDepth/zm))-2)*(((zm-obsTherm.Therm)/zm)**2))-((2*(MeanDepth/zm))-1)*(((zm-obsTherm.Therm)/zm)**3))/((MeanDepth/zm))
        V = 1-V
    else:
        
        # Reset index to have the original index of each data frame to then extract the index in which both match
        NewHypso.reset_index(inplace = True)
        NewHypso['depth'] = round(NewHypso['depth'], 1)

        obsTherm.reset_index(inplace = True) # x2 reset in StratData since the 1st takeout sampledate from index and
        obsTherm.reset_index(inplace = True) # 2nd reset to pass the original index as a column to then match

        # Mask to find  the index position in which the Thermocline depth match with depth of hypso -> index_x
        mask =  pd.merge(obsTherm,NewHypso, left_on='Therm', right_on='depth', how = 'left')['index_y']
        V = NewHypso.Hypso_cum.iloc[mask].values
    
    # Parameter Initilization
    Qout = np.asarray(Inflow['Discharge.m3.d'])[0:Tdays+1]
    LoadT = np.asarray(Inflow['P.g.day'])[0:Tdays+1]
    Cloadpp = (LakeConfig.LoadAsAdsorbed).item() # .item() To get scalar ;) 
    Cburial =  (LakeConfig.Burial).item() # .item() To get scalar ;) 
    Strat = np.asarray(StratData.Strat)


    CoefR = np.asarray(ThetaR**(StratData.HypoTemp - TbaseR)) # Arrhenius coefficient for recycling
    CoefS = np.asarray(ThetaS**(StratData.EpiTemp - TbaseS)) # Arrhenius coefficient for sedimentation (assuming phytos are sinking)
    
    # Calculate epi and hypo volume vectors, which should always add up to lake volume
    HypV = (LakeVolume * (1-V))# hypolimnetic volume time series
    EpiV = LakeVolume-HypV
    # Change in epilimnetic volume (m3), used for entrainment
    # where a +change means transfer of volume from hypo to epi
    #dEpiV = np.diff(EpiV)
    dEpiV = np.insert(np.diff(EpiV),0,0)
    #dEpiV  = EpiV.diff()

      #Initial Lake P Concentration
    LakePConc = 0.0860 # obsCDEpi$totpuf_sloh #g/m^3

    Pepi = np.asarray([np.nan]*(len(Ndays)))
    Pepi[0] = (LakePConc * EpiV[1]) # g of P in water

    Phyp = np.asarray([np.nan]*(len(Ndays)))
    Phyp[0] = (LakePConc * HypV[1])

    Psed = np.asarray([np.nan]*(len(Ndays)))
    Psed[0] = (SedimentArea * SedimentPConc) #g of P in sediment


    #######################                    To be filled by the model                   ################

    cfsedP =np.asarray([np.nan]*len(Ndays))
    R =np.asarray([np.nan]*len(Ndays))
    E =np.asarray([np.nan]*len(Ndays))
    Sed =np.asarray([np.nan]*len(Ndays))
    B =np.asarray([np.nan]*len(Ndays))
    I =np.asarray([np.nan]*len(Ndays))
    Entr =np.asarray([np.nan]*len(Ndays))
    TPsed  =np.asarray([np.nan]*len(Ndays))
    
   # Mechanistic Model #
    for i in Ndays[1:]:           
        #Entrainment
        if dEpiV[i] == 0:        
            Entr[i] = 0

        elif dEpiV[i] >0:        
            Entr[i] = (dEpiV[i]/HypV[i-1])*Phyp[i-1]

        else:
            Entr[i] = (dEpiV[i]/EpiV[i-1])*Pepi[i-1]

        # Recycling 
        if Psed[i-1] <=0:
            cfsedP[i]= 0
        else:
            TPsed[i] = Psed[i-1]*(1/density)*(1/SedV)*(1/0.001)
            cfsedP[i]= (a+b*TPsed[i])

        R[i] = LakeArea*cfsedP[i]*(1/1000)*CoefR[i-1] # 
        E[i] = Qout[i-1]*(Pepi[i-1]/EpiV[i-1])
        I[i] = (1-Cloadpp)*LoadT[i-1] # ? 
        B[i] = Cburial*(1/365)*(1/1000)*Psed[i-1]/SedDepth
        Sed[i] = Csed*Pepi[i-1]*CoefS[i-1] #?

        # Update

        if Strat[i-1] == True and Strat[i] == False: # Fall turnover
            Entr[i] = 0

            Pepi[i] = (V[i])*(Phyp[i-1]+Pepi[i-1]) + I[i] + R[i]*(V[i-1])*EpiFactor + Entr[i] - Sed[i] -E[i]
            Phyp[i] = (1-V[i])*(Phyp[i-1]+Pepi[i-1]) + R[i]*(1-V[i-1]) - Entr[i]
            Psed[i] = Psed[i-1] + Sed[i] -(R[i]*(V[i-1])*EpiFactor +R[i]*(1-V[i-1])) + LoadT[i-1]*Cloadpp - B[i]

        elif Strat[i-1] == False and Strat[i] == True: # Spring Stratification        
           
            Entr[i] = 0

            Pepi[i] = (V[i])*(Phyp[i-1]+Pepi[i-1]) + I[i] + R[i]*(V[i-1])*EpiFactor + Entr[i] - Sed[i] -E[i]
            Phyp[i] = (1-V[i])*(Phyp[i-1]+Pepi[i-1]) + R[i]*(1-V[i-1]) - Entr[i]
            Psed[i] = Psed[i-1] + Sed[i] -(R[i]*(V[i-1])*EpiFactor +R[i]*(1-V[i-1])) + LoadT[i-1]*Cloadpp - B[i]

        elif Strat[i-1] == False and Strat[i] == False: # Winter
            R[i]   = 0
            Entr[i] = 0

            Pepi[i] = (V[i])*(Phyp[i-1]+Pepi[i-1]) + I[i] + R[i]*(V[i-1])*EpiFactor + Entr[i] - Sed[i] -E[i]
            Phyp[i] = (1-V[i])*(Phyp[i-1]+Pepi[i-1]) + R[i]*(1-V[i-1]) - Entr[i]
            Psed[i] = Psed[i-1] + Sed[i] -(R[i]*(V[i-1])*EpiFactor +R[i]*(1-V[i-1])) + LoadT[i-1]*Cloadpp - B[i]

        elif Strat[i-1] == True and Strat[i] == True: # Summer        
            Pepi[i] = Pepi[i-1] + I[i] + R[i]*(V[i-1])*EpiFactor + Entr[i] - Sed[i] -E[i]
            Phyp[i] = Phyp[i-1] + R[i]*(1-V[i-1]) - Entr[i]
            Psed[i] = Psed[i-1] + Sed[i] -(R[i]*(V[i-1])*EpiFactor +R[i]*(1-V[i-1])) + LoadT[i-1]*Cloadpp - B[i] 
        
    return(Pepi,Phyp,Psed)
