import tabula
import dateutil
import numpy as np
import pandas as pd
from flask import Flask, request,render_template
import pickle
import xml.etree.ElementTree as ET
from datetime import datetime,timedelta

app = Flask(__name__)

#This function read cibil xm file and return data:
def CIBIL(file):
    # create element tree object 
    tree = ET.parse(file) 
    # get root element 
    root = tree.getroot() 
    context = root.find('ContextData')
    
    cibil = context.getchildren()[0].find('Applicants').find('Applicant').find('DsCibilBureau')
    credit_report = cibil.find('Response').find('CibilBureauResponse').find('BureauResponseXml').find('CreditReport')
    if credit_report:
        name_segment = credit_report.findall('NameSegment')[0]
        
        try:
            id_segment = credit_report.findall('IDSegment')[0]
        except:
            id_segment = '-'
            
        try:    
            tele_segment = credit_report.findall('TelephoneSegment')[0]
        except:
            tele_segment = '-'
        
        try:
            email_segment = credit_report.findall('EmailContactSegment')[0]
        except:
            email_segment = '-'
        
        try:
            addresses = credit_report.findall('Address')[0]
        except:
            addresses = np.nan
            
        try:
            score_segment = credit_report.find('ScoreSegment')
        except:
            score_segment = np.nan
            
        
        if len(credit_report.findall('Account'))>=1:
            accounts = credit_report.findall('Account')[0]
        else:
            accounts = np.nan
            
        enquiries = credit_report.findall('Enquiry')
        
        #NameSegment:
        name1 = name_segment.find('ConsumerName1').text
        
        try:
            name2 = name_segment.find('ConsumerName2').text
        except:
            name2 = np.nan
            
        dob = name_segment.find('DateOfBirth').text
        gender = name_segment.find('Gender').text
        if int(gender) == 1:
            gender = 'Female'
        else:
            gender = 'Male'
            
        #IDSegment:
        if id_segment != '-':
            pan_no = id_segment.find('IDNumber').text
        else:
            pan_no = '-'
            
        #Telephone Segment:
        if tele_segment != '-':
            phone_no = tele_segment.find('TelephoneNumber').text
        else:
            phone_no = '-'
        
        #Email Segment:
        if email_segment != '-':
            email = email_segment.find('EmailID').text
        else:
            email = np.nan
            
        #Score Segement:
        if score_segment.find('Score').text == '000-1':
            cibilscore = int(score_segment.find('Score').text.split('-')[-1])
    
        else:
            cibilscore = int(score_segment.find('Score').text)
            
        #Address Segment:
        a1 = addresses.find('AddressLine1').text
        try:
            a2 = addresses.find('AddressLine2').text
        except:
            a2 = ''
    
        address = a1+', '+a2
        try:
            city = a2.split()[-1]
        except:
            city = '-'
    
        pin = addresses.find('PinCode').text
        
        #Account Segment:
        try:
            details = accounts.find('Account_NonSummary_Segment_Fields')
        except:
            details = '-'
            
        try:
            ac_no = details.find('AccountNumber').text
        except:
            ac_no = np.nan
            
        try:
            open_date = details.find('DateOpenedOrDisbursed').text
        except:
            open_date = np.nan
            
            
        try:
            last_date = details.find('DateOfLastPayment').text
        except:
            last_date = np.nan
        
        try:
            emi = details.find('EmiAmount').text
        except:
            emi = np.nan
    
        try:
            amount = details.find('HighCreditOrSanctionedAmount').text
        except:
            amount = np.nan
    
        try:
            balance = details.find('CurrentBalance').text
        except:
            balance = np.nan
            
        try:
            overdue = details.find('AmountOverdue').text
        except:
            overdue = np.nan
            
        try:
            interest = details.find('RateOfInterest').text
        except:
            interest = np.nan
            
        try:
            tenure = details.find('RepaymentTenure').text
        except:
            tenure = np.nan
            
        try:
            collateral_Value = details.find('ValueOfCollateral').text
        except:
            collateral_Value = np.nan
            
        try:
            due_days1 = details.find('PaymentHistory1').text
        except:
            due_days1 = np.nan
            
        try:
            due_days2 = details.find('PaymentHistory2').text
        except:
            due_days2 = np.nan
            
        #EnquirySegment:
        total_no_enquiries = len(enquiries)
        
        if total_no_enquiries >= 1:
            enquiry = enquiries[0]
        else:
            enquiry = np.nan
            
        try:
            last_enq_purpose = enquiry.find('EnquiryPurpose').text
        except:
            last_enq_purpose = np.nan
            
        try:
            last_enq_date = enquiry.find('DateOfEnquiryFields').text
        except:
            last_enq_date = np.nan
    
        try:
            last_enq_amt = enquiry.find('EnquiryAmount').text
        except:
            last_enq_amt = np.nan
            
        whole_data = [name1, 
                        name2,
                        dob,
                        gender,
                        pan_no,
                        phone_no,
                        email,
                        cibilscore,
                        address,
                        city,
                        pin,
                        ac_no,
                        open_date,
                        last_date,
                        amount,
                        balance,
                        overdue,
                        interest,
                        tenure,
                        emi,
                        collateral_Value,
                        due_days1,
                        due_days2,
                        total_no_enquiries,
                        last_enq_date,
                        last_enq_purpose,
                        last_enq_amt]    
        
        try:
            #Finding Enquiries per Month:
            enq = []
            if len(enquiries)>1:
                for i in range(len(enquiries)):
                    row = []
                    row.append(enquiries[i].find('DateOfEnquiryFields').text)
                    row.append(enquiries[i].find('EnquiryPurpose').text)
                    row.append(enquiries[i].find('EnquiryAmount').text)
                    enq.append(row)
            enq = pd.DataFrame(enq,columns=["Date","Purpose","Amount"])
            def fun(dat):
                stop = datetime.strptime(dat, "%d%m%Y").date()
                return stop
            enq['Date'] = enq['Date'].apply(fun)
            enq['Date'] = enq['Date'].astype(str)
            enq['Date'] = enq['Date'].apply(dateutil.parser.parse, dayfirst=True)
        except:
            pass
    
    else:
        whole_data = None
        
    return whole_data,enq

#This is a function to read the pdf bank statement of HDFC Bank:
def HDFC_PDF(file):
    """
    This function wil parse the HDFC PDF Statement.
    This function will return the average bank balance per month.
    """
    #Read PDF file:
    tables = tabula.read_pdf(file,pages='all')

    #Combining all tables:
    table = []
    for i in range(len(tables)):
        s1 = tables[i].values.tolist()
        table.extend(s1)


    #Removing unwanted columns:
    ex = []
    for i in range(len(tables)):
        if tables[i].shape[1] == 7:
            ex.extend(tables[i].values.tolist())
        elif tables[i].shape[1] == 6:
            table = tables[i].values.tolist()
            for i in table:
                i.append(i[5])
                i[5] = np.nan
                ex.append(i)

        elif tables[i].shape[1] == 8:
            table = tables[i].values.tolist()
            for i in table:
                del i[2]
                ex.append(i)

    #Creating dataframe:
    df = pd.DataFrame(ex,columns=['Date', 'Narration', 'Chq./Ref.No.', 'Value Dt', 'Withdrawal Amt.','Deposit Amt.', 'Closing Balance'])

    #Removing rows which having date is null:
    df = df[~df['Date'].isnull()]

    #Parsing Closing Price
    df["Closing Balance"] = df["Closing Balance"].astype(str)

    #Converting dataset into List:
    l1 = df.values.tolist()

    #Handiling Closing Balance column:
    final = []
    for i in l1:
        splits = (i[-1].split())
        if (len(splits)>1):
            i[-2] = splits[0]
            i[-1] = splits[1]
            final.append(i)
        else:
            final.append(i)

    #Creating dataframe:
    final = pd.DataFrame(final,columns=['Date', 'Narration', 'Chq/Ref.No', 'Value Dt', 'Withdrawal Amt','Deposit Amt', 'Closing Balance'])

    #Calculating check bounce:
    bal_list = final.iloc[:,-1].values.tolist()
    
    val = []
    for j in bal_list:
        val.append(''.join(j.split(','))) 
    val = [float(i) for i in val]
    
    bounce = 0
    for i in val:
        try:
            if i<0:
                bounce = bounce + 1
        except:
            continue
    
    #Parsing the date fields:
    final['Date'] = final['Date'].apply(dateutil.parser.parse, dayfirst=True)
    final['Value Dt'] = final['Value Dt'].apply(dateutil.parser.parse, dayfirst=True)

    #Paring prices:
    final['Closing Balance'] = final['Closing Balance'].astype(str)
    col = ['Closing Balance']
    for i in col:
        val = []
        for j in final[i]:
            val.append(''.join(j.split(',')))
        final[i] = val

    #TypeCasting Closing Balance:
    col = ['Closing Balance']
    for i in col:
        final[i] = pd.to_numeric(final[i],errors='coerce')

    #Group by operation to close price:
    group = final.groupby(pd.Grouper(key='Date',freq='1M'))

    #Filtering close balance per month:
    balance_month = []
    for i in group:
        a = i[1]
        balance_month.append(a['Closing Balance'].iloc[-1])

    #Closing Balance Per Month:
    return np.average(balance_month),bounce


#Loading a model:
model = pickle.load(open('model_tw_sig.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/back',methods=['POST','GET'])
def back():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    predict_request = []
    res = []
    
    #Uploading file:
    cibil_file = request.files['cibil']   
    destination = cibil_file
    cibil_data,enq = CIBIL(destination)
    
    status = request.form["martial_status"]
    married = {2750:"Married",2751:"Un Married"}
    predict_request.append(status)
    res.append(married.get(int(status)))
    
    dep = request.form["dependants"]
    predict_request.append(dep)
    res.append(dep)
    
    resi = request.form["residence"]
    residence = {2755:"Own",2756:"Rent"}
    predict_request.append(resi)
    res.append(residence.get(int(resi)))
    
    year = request.form["staying_year"]
    predict_request.append(year)
    res.append(year)
    
    #Uploading file:
    file = request.files['file']
    filename = file.filename
    extn = filename.split('.')[-1]   
    destination = file
  
    #Checking for extension of file: 
    if (extn.casefold() == 'pdf'):
        #Returned a result from a function calling:
        clobal,bounce =  HDFC_PDF(destination)
    
    if (extn.casefold() == 'xls'):
        #Loading dataset:
        df = pd.read_excel(destination)
        
        #Fetching transactions only:
        row_no = 0
        for i in df.iloc[:,0]:
            if i == 'Date':
                df = df.iloc[row_no:]
                break
            row_no = row_no+1
        
        #Set a features name:
        df.columns = ['Date', 'Narration', 'Chq./Ref.No.', 'Value Dt', 'Withdrawal Amt.','Deposit Amt.', 'Closing Balance']
        
        #Reset the Index:
        df.reset_index(drop=True, inplace=True)
        
        #Dropping first two records:
        df.drop([0,1],axis=0,inplace=True)
        
        #Reset the Index:
        df.reset_index(drop=True, inplace=True)
        
        row_no = 0
        for i in df['Date']:
            if len(str(i)) != 8:
                df = df.iloc[:row_no]
                break
            row_no = row_no + 1
            
        bal_list = df.iloc[:,-1].values.tolist()
        
        bounce = 0
        for i in bal_list:
            try:
                if i<0:
                    bounce = bounce + 1
            except:
                continue
            
        # Parsing date:
        df['Date'] = df['Date'].apply(dateutil.parser.parse, dayfirst=True)
        table = df
        
        #Group by operation to find opening and close price:
        group = table.groupby(pd.Grouper(key='Date',freq='1M'))
        
        #Filtering open and close balance per month:
        balance_month = []
        for i in group:
            a = i[1]
            balance_month.append(a['Closing Balance'].iloc[-1])
        
        clobal = (np.average(balance_month))
   
    predict_request.append(clobal)
    res.append("{:,}".format(int(clobal)))

    asset = request.form["assetvalue"]
    predict_request.append(asset)
    res.append("{:,}".format(int(asset)))
    
    cat = request.form["productcat"]
    prod_cat = {1784:"LOAN AGAINST PROPERTY",
            926:"CAR",
            912:"MULTI UTILITY VEHICLE",
            945:"VIKRAM",
            1402:"TRACTOR",
            1373:"USED VEHICLES",
            1672:"TIPPER",
            1664:"FARM EQUIPMENT",
            1541:"TWO WHEELER",
            634:"INTERMEDIATE COMMERCIAL VEHICLE",
            527:"HEAVY COMMERCIAL VEHICLE",
            528:"CONSTRUCTION EQUIPMENTS",
            529:"THREE WHEELERS",
            530:"LIGHT COMMERCIAL VEHICLES",
            531:"SMALL COMMERCIAL VEHICLE",
            738:"MEDIUM COMMERCIAL VEHICLE",
            783:"BUSES"}
    predict_request.append(cat)
    res.append(prod_cat.get(int(cat)))
    
    brand = request.form["brand"]
    brand_type = {1:"Others",
                  1360:"HONDA",
                  1542:"HERO", 
                  1544:"HMSI",
                  1547:"YAMAHA",
                  1546:"SUZUKI",
                  1647:"TVS",
                  1549:"ROYAL ENFIELD"
                  }
    predict_request.append(brand)
    res.append(brand_type.get(int(brand)))
    
    indus = request.form["industrytype"]
    ind_cat = {1782:"SALARIED",1783:"SELF EMPLOYEED",603:"AGRICULTURE",
     604:"PASSENGER TRANSPORTATION",605:"CONSTRUCTION",875:"INFRASTRUCTURE",
     876:"CEMENT",877:"OIL AND GAS",878:"GOVERNMENT CONTRACT",879:"OTHERS",658:"MINE"}
    predict_request.append(indus)
    res.append(ind_cat.get(int(indus)))
    
    tenure = request.form["tenure"]
    predict_request.append(tenure)
    res.append(tenure)
    
    instal = request.form["instalcount"]
    predict_request.append(instal)
    res.append(instal)
    
    chasasset = request.form["chasasset"]
    predict_request.append(chasasset)
    res.append("{:,}".format(int(chasasset)))
    
    chasinitial = request.form["chasinitial"]
    predict_request.append(chasinitial)
    res.append("{:,}".format(int(chasinitial)))
    
    chasfin = int(chasasset) - int(chasinitial)
    predict_request.append(chasfin)
    res.append("{:,}".format(int(chasfin)))
    
    fininter = request.form["finaninterest"]
    predict_request.append(fininter)
    res.append(fininter)
    
    interestamount = (int(chasfin)*(int(tenure)/12)*(float(fininter)))/100
    emi = (int(chasfin)+int(interestamount))/int(tenure)
    predict_request.append(int(emi))
    res.append("{:,}".format(int(emi)))
    
    inflow = request.form["totinflow"]
    predict_request.append(inflow)
    res.append("{:,}".format(int(inflow)))
    
    #Cibil Score from xml data:
    if str(cibil_data[7]) != 'nan':
        cibil = cibil_data[7]
    else:
        cibil = 0
    predict_request.append(cibil)
    res.append(cibil)
    
    age = request.form["age"]
    predict_request.append(age)
    res.append(age)
    
    #############################################
    if int(age) >= 60:
        age_score = 1
    elif int(age) >= 50:
        age_score  = 2
    elif int(age) >= 40:
        age_score = 3
    elif int(age) >= 30:
        age_score = 5
    else:
        age_score = 0   
    predict_request.append(age_score)
    res.append(age_score)
    
    
    if int(resi) == 2755:
        TOR_Score = 5
    else:
        TOR_Score = 0
    predict_request.append(TOR_Score)
    res.append(TOR_Score)
    
    if int(year) >= 3:
        NYS = 5
    elif int(year) >= 2:
        NYS = 3
    elif int(year) <= 1:
        NYS = 0
    else:
        NYS = 2
    predict_request.append(NYS)
    res.append(NYS)
    
    if int(dep) >= 2:
        Dependant_Score = 3
    else:
        Dependant_Score = 5
    predict_request.append(Dependant_Score)
    res.append(Dependant_Score)
    
    
    l2v = int((int(chasfin)*100)/int(chasasset))
    if l2v <= 60:
        L2V_Score = 5
    elif l2v <= 80:
        L2V_Score = 3
    elif l2v <=90:
        L2V_Score = 2
    else:
        L2V_Score = 0
    predict_request.append(L2V_Score)
    res.append(L2V_Score)
    
    #over 3years(5) 2Years(3) one year(2) others(0)
    stability = request.form["stability"]
    stab_cat = {
            1 :"Salaried with over 3 years",
            2 :"Salaried with over 2 years",
            3 :"Salaried with over 1 year and above",
            4 :"Others"}
    res.append(stab_cat.get(int(stability)))
    if int(stability) == 1:
        stability_score = 5
    elif int(stability) == 2:
        stability_score = 3
    elif int(stability) == 3:
        stability_score = 2
    elif int(stability) == 4:
        stability_score = 0
    predict_request.append(stability_score)
    res.append(stability_score)
    
    if int(chasfin) < int(inflow):
        salary_score = 5
    elif (int(inflow)/int(chasfin)) >= 50:
        salary_score = 4
    elif (int(inflow)/int(chasfin)) >= 25:
        salary_score = 2
    else:
        salary_score = 1
    predict_request.append(salary_score)
    res.append(salary_score)
    
    ctc = int(inflow)*12
    if (ctc == int(chasfin)):
        Borrwing_Score = 5
    elif (2*(ctc) == int(chasfin)):
        Borrwing_Score = 3
    else:
        Borrwing_Score = 0
    predict_request.append(Borrwing_Score)
    res.append(Borrwing_Score)
    
    #Cibil Score:
    if int(cibil) == 700:
        CIBIL_Score = 3
    elif int(cibil) > 700:
        CIBIL_Score = 5
    else:
        CIBIL_Score = 1
    predict_request.append(CIBIL_Score)
    res.append(CIBIL_Score)

    dpd60_point = False
    dpd30_point = True
    #Over 60DPD inall(0) Suit filed(0)60DPD in one forth of loans by value(3)less than 30DPD in all(5):
    for history in cibil_data[21:23]:
        if str(history) != 'nan':
            l1 = history
            loc = []
            for i in range(0,len(l1),3):
                loc.append(i)
    
            due = []
            i= 0
            j = i+1   
            count =0
            while count+1<len(loc):
                count = count+1
                due.append(l1[loc[i]:loc[j]])
                i = i+1
                j = j+1 
            due.append(l1[loc[j-1]:])
    
            for i in range(len(due)):
                try:
                    due[i] = int(due[i])
                except:
                    due[i] = due[i]     
    
            for i in due:
                try:
                    if i >= 60:
                        dpd60_point = True
                        break
                except:
                    continue
    
            for i in due:
                try:
                    if (i == 'XXX') or (i <= 30):
                        continue
                    else:
                        dpd30_point = False
                        break
                except:
                    continue 
        else:
            break
    
    print(dpd30_point)    
    if dpd30_point:
        predict_request.append(5)
        res.append(5)
    elif dpd60_point:
        res.append(0)
        predict_request.append(0)
    else:
        res.append(0)
        predict_request.append(0)
    
    #Similar products in three months(0) else(1):
    enquiry_point = False
    if len(enq) > 1:
        enq_list = enq.values.tolist()
        for i in enq_list:
            if i[0] > datetime.today()-timedelta(90):
                if i[1] == '13':
                    enquiry_point = True
                    break
            
    if enquiry_point:
        predict_request.append(0)
        res.append(0)
    else:
        res.append(5)
        predict_request.append(5)
    
    #Account over 2yearvintage(5) one year and above(3)over6 months(2)0thers(0):
    bank = request.form["bank_detail"]
    bank_cat = {1 :"Account over 2year",
             2:"One year and above",
             3:"Over 6 months",
             4 :"Others"}
    res.append(bank_cat.get(int(bank)))
    
    if int(bank) == 1:
        bank_score = 5
    elif int(bank) == 2:
        bank_score = 3
    elif int(bank) == 3:
        bank_score = 2
    if int(bank) == 4:
        bank_score = 0
    predict_request.append(bank_score)
    res.append(bank_score)
    

    if int(clobal) >=  (2*int(emi)):
        avg_bal_score = 5
        
    elif int(clobal) >=  int(emi):
        avg_bal_score = 4
        
    elif int(clobal) >= ((int(emi)/100)*90):
        avg_bal_score = 3
    else:
        avg_bal_score = 1
    predict_request.append(avg_bal_score)
    res.append(avg_bal_score)
    
    #5	NIL(5)  others(0) checkbounce
    if bounce > 0:
        predict_request.append(0)
        res.append(0)
    else:
        res.append(5)
        predict_request.append(5)
    
    #############################################    
    gender_dict = {'M':[0,1],'F':[1,0]}
    cate = request.form["gender"]
    if cate == 'M':
        res.append('Male')
    else:
        res.append('Female')
        
    res.append(request.form["pan"])
    
    geo_cat = {1 :"Less than 15 Km",
               2 :"More than 15 Km"}
    res.append(geo_cat.get(int(request.form["geo"])))
        
    predict_request.extend(gender_dict.get(cate))
    predict_request = list(map(float,predict_request))
    predict_request = np.array(predict_request)
    prediction = model.predict_proba([predict_request])[0][-1]
    output = int((1 - prediction)*100)
    if output < 50:
        condition = 'Risky'
    if output >= 50 and output <= 69:
        condition = 'Barely Acceptable'
    if output >= 70 and output <=89:
        condition = 'Medium'
    if output >= 90 and output <= 99:
        condition = 'Good'
    if output == 100:
        condition = 'Superior'
    print('######################################')
    print(output)
    print('######################################')
    return render_template('resultpage.html', prediction_text=output,data=res,status=condition,info=cibil_data)

if __name__ == "__main__":
    app.run(debug=True)
