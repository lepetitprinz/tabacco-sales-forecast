import sys
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from urllib.request import urlopen

# -------------------------------------------- #
# Define Functions
# -------------------------------------------- #

def get_xml_data(xtree, format: int):
    """
    :param xtree: XML Tree
    :param format:
            0: ovsea / foreign_stat
            1: foreign_avg
    :return:
    """
    if format == 1:
        rows = []
        for node in xtree[1][0]:
            n_age = node.find("age").text
            n_ageCd = node.find("ageCd").text
            n_num = node.find("num").text
            n_port = node.find("port").text
            n_portCd = node.find("portCd").text
            n_rnum = node.find("rnum").text
            n_sex = node.find("sex").text
            n_sexCd = node.find("sexCd").text
            n_ym = node.find("ym").text

            rows.append({"age": n_age, "ageCode": n_ageCd, "num": n_num, "port": n_port, "portCode": n_portCd,
                         "year": n_rnum, "sex": n_sex, "sexCode": n_sexCd, "yymm": n_ym})

        return rows

    else:
        rows = []
        for node in xtree[1][0]:
            n_natCd = node.find("natCd").text
            n_natKorNm = node.find("natKorNm").text
            n_sojAvg = node.find("sojAvg").text
            n_sojTot = node.find("sojTot").text
            n_ym = node.find("ym").text

            rows.append({"nationCode": n_natCd, "nationKorea": n_natKorNm, "average": n_sojAvg, "total": n_sojTot, "yyyymm": n_ym})

        return rows

def get_prep_df(rows: list, format: int):
    if format == 1:
        temp = pd.DataFrame(rows)
        temp = temp[temp['ageCode'] > '20']  # 20세 이상
        temp = temp[temp['ageCode'] != '99']  # 승무원 제외
        temp = temp[(temp['portCode'] == 'IA') | (temp['portCode'] == 'GP') | (temp['portCode'] == 'CJ')]
        temp['num'] = temp['num'].astype(np.int64)

    else:
        temp = pd.DataFrame(rows)
        temp['average'] = temp['average'].astype(np.float64)

    return temp

def get_stats_df(yyyymm_list: list, url: str, format: int):

    nums = []

    for yyyymm in yyyymm_list:
        url_opt = f'?YM={yyyymm}&numOfRows={num_of_row}&serviceKey={servicekey}'
        url_fin = url + url_opt
        response = urlopen(url_fin).read()
        xtree = ET.fromstring(response)

        rows = get_xml_data(xtree=xtree, format=format)
        temp = get_prep_df(rows=rows, format=format)
        if format == 1:
            nums.append({'yyyymm': yyyymm, 'num_of_people': temp['num'].sum()})
        else:
            nums.append({'yyyymm': yyyymm, 'avg_retention_time': round(temp['average'].mean())})

    df = pd.DataFrame(nums)
    if format == 1:
        df = df[['yyyymm', 'num_of_people']]
    else:
        df = df[['yyyymm', 'avg_retention_time']]
    return df

# -------------------------------------------- #
# Main
# -------------------------------------------- #
url_get_ovsea = 'http://openapi.tour.go.kr/openapi/service/EdrcntTourismStatsService/getOvseaTuristStatsList'
url_get_foreign_stat = 'http://openapi.tour.go.kr/openapi/service/EdrcntTourismStatsService/getForeignTuristStatsList'
url_get_foreign_avg = 'http://openapi.tour.go.kr/openapi/service/EdrcntTourismStatsService/getForeignTuristAvrgList'
servicekey = 'M3aNb9idrJyu5WxbqPd5RNByfeov%2BP%2FcfyxMlW%2BJGmN3ZH5G%2FXUsxRJ%2Bby%2Fe0S892NS4Xo%2BBjdkfRuqGjWuxkQ%3D%3D'

num_of_row = 10000

# Set date range
date_from = '20200301'  # 20050101
date_to = '20200501'    # 20200601
yyyymm_list = [day.strftime('%Y%m') for day in pd.date_range(date_from, date_to)]
yyyymm_list = list(set(yyyymm_list))    # 중복 제거
yyyymm_list.sort()

# oversea = get_stats_df(yyyymm_list=yyyymm_list, url=url_get_ovsea, format=1)
# foreign_stat = get_stats_df(yyyymm_list=yyyymm_list, url=url_get_foreign_stat, format=1)
foreign_avg = get_stats_df(yyyymm_list=yyyymm_list, url=url_get_foreign_avg, format=2)
print(foreign_avg)
