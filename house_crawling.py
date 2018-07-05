import xlsxwriter
import requests
import os,re
from lxml import html
import time
class house_crawl():
    def __init__(self):
        self.row=0
        self.workbook = xlsxwriter.Workbook(self.getdesktoppath() + '\\house_price.xlsx')
        self.worksheet1 = self.workbook.add_worksheet()
        self.worksheet1.write_row(self.row,0,['address','describe1','describle2','price','avg_price'])
        self.row+=1
    def __call__(self, url):
        self.page_crawl(url)
    def close(self):
        self.workbook.close()
    def page_crawl(self,url):
        page=requests.get(url)
        tree=html.fromstring(page.text)
        for house in tree.xpath('//*[@class="house-item clearfix"]'):
            address=house.xpath('div[1]/p[3]/text()')[0]
            address=re.sub('\r|\n| ','',address)
            #print(address)浦东-潍坊崂山路571弄（旧址崂山东路571弄）
            d1=[]
            for i in house.xpath('div[1]/p[1]/child::*'):
                d1.append(i.xpath('text()')[0])
            st=','
            d1=st.join(d1)
            # print(d1)东欣高层,3室1厅,70平
            d2=house.xpath('div[1]/p[2]/text()')
            stringt=','
            d2=stringt.join(d2)
            d2 = re.sub('\r|\n| ','',d2)
            #print(d2)南,中层,中装,1991年
            price=house.xpath('div[2]/p[1]/text()')[0]
            #print(price)580万
            avg_price=house.xpath('div[2]/p[2]/text()')[0]
            if '元/平' not in avg_price:
                avg_price = house.xpath('div[2]/p[3]/text()')[0]
            #print(avg_price)83400元/平
            self.worksheet1.write_row(self.row, 0, [address,d1,d2,price,avg_price])
            self.row+=1
    def getdesktoppath(self):
        return os.path.join(os.path.expanduser("~"), 'Desktop')
hc=house_crawl()
page_num=1546
for i in  range(1,page_num+1):
    print('正在爬去第{}页'.format(i))
    url = 'http://sh.centanet.com/ershoufang/g{}/?sem=baidu_ty'.format(i)
    try:
        hc(url)
    except:
        hc.close()
    time.sleep(0.2)
hc.close()


