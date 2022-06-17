
import pandas as pd
import numpy as np
import os

def process_general_info(data_ad):
    print ("min time {}, max time {}".format(data_ad['RequestDate'].min(),data_ad['RequestDate'].max()))
    print ("total data shape: {}".format(data_ad.shape))
    print ("by type distribution: sp {}, sb{} sd{} sbv{} nan{}".format(
           len(data_ad[data_ad['AdType']=='SP']),\
           len(data_ad[data_ad['AdType']=='SB']),\
           len(data_ad[data_ad['AdType']=='SD']),\
           len(data_ad[data_ad['AdType']=='SBV']),\
           len(data_ad[data_ad['AdType']=='nan'])
          ))
    #print ("total asin number: ", data_ad['Asin'].nunique())
    print ("total ID number: ", data_ad['ID'].nunique())
    print ("total asin_k number: ", data_ad['asin_k'].nunique())
    return

def read_data_merge_save(data_directory):
    # daily ads data
    df_adv_daily = pd.read_excel(os.path.join(data_directory,'ads-daily.xlsx'),
                                 engine="openpyxl")
    # product info
    product_info_1 = pd.read_excel('../data-advertisement-datav2/数据提供-广告：AmazonAllListingReport.xlsx',
                                   engine="openpyxl")
    product_info_2 = pd.read_excel('../data-advertisement-datav2/数据提供-广告：AmazonMachingProductForSk....xlsx',
                                   engine="openpyxl")
    # sales info
    sales_daily = pd.read_excel('../data-advertisement-datav2/数据提供-广告：日销数据.xlsx',
                                engine="openpyxl")
    # campaign data
    df_campaign = pd.read_excel('../data-advertisement-datav2/数据提供-广告：活动数据.xlsx',
                                engine="openpyxl")
    # keyword data
    data_keyword_old_ranking = pd.read_csv('../aws-v1/asin旧关键词排名.csv',
                                           encoding='gbk')
    data_keyword = pd.read_csv('../aws-v1/asin关键词排名.csv')

    #process daily and describe
    process_general_info(df_adv_daily)

    # convert time type
    sales_daily['time'] = pd.to_datetime(sales_daily['日期'], format='%Y-%m-%d')

    ####  join data
    # join sales
    df_daily = pd.merge(df_adv_daily, sales_daily, left_on=['RequestDate', 'asin_k'], right_on=['time', 'asin_k'],
                        how='inner')
    # join sku information
    # join sales
    df_daily = pd.merge(df_daily, product_info_1[['ItemName', 'ItemDescription', 'asin_k']], left_on=['asin_k'],
                        right_on=['asin_k'], how='left')
    df_daily = pd.merge(df_daily, product_info_2[
        ['Binding', 'Brand', 'Label', 'ListPriceAmount', 'Color', 'asin_k', 'ProductGroup', 'ProductTypeName']],
                        left_on=['asin_k'], right_on=['asin_k'], how='left')
    # join campaign data
    # only keep campain data tha is not cancelled
    df_campaign_active = df_campaign[df_campaign['campaign_status'] != 'Cancelled']
    # map to multi-rows
    df_campaign_active['DATE'] = [pd.date_range(s, e, freq='d') for s, e in
                                  zip(pd.to_datetime(df_campaign_active['site_start_time']),
                                      pd.to_datetime(df_campaign_active['site_end_time']))]

    df_campaign_active = df_campaign_active.explode('DATE').drop(['site_start_time', 'site_end_time'], axis=1)
    # join back to sales
    df_daily = pd.merge(df_daily, df_campaign_active[
        ['campaign_type', 'sales_price', 'campaign_price', 'discount', 'DATE', 'asin_k']],
                        left_on=['RequestDate', 'asin_k'], right_on=['DATE', 'asin_k'], how='left')


    return df_daily







