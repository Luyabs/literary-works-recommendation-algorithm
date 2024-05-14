import random
import re
import string
import urllib.error
import urllib.request

from bs4 import BeautifulSoup
from tqdm import tqdm

from mysql_connector.mysql_connector_keep_alive import MysqlConnectorKeepAlive


# 获取html
def get_html(url):
    head = {
        "User-Agent": "Mozilla / 5.0(Windows NT 10.0; Win64; x64) AppleWebKit / 537.36(KHTML, like Gecko) Chrome / 80.0.3987.122  Safari / 537.36",
        "Cookie": "bid=%s" % "".join(random.sample(string.ascii_letters + string.digits, 11))
    }
    request = urllib.request.Request(url, headers=head)
    try:
        response = urllib.request.urlopen(request)
        html = response.read().decode("utf-8")
        return html
    except urllib.error.URLError as e:
        if hasattr(e, "code"):
            print(e.code)
        if hasattr(e, "reason"):
            print(e.reason)


# 获取网页中的目标信息
def get_info(html):
    if html is None:
        print('html is None')
        return '', '', '', ''

    soup = BeautifulSoup(html, 'html.parser')

    try:
        # 提取作者和出版社信息
        info_html = soup.find('div', id='info').find_all('a')
        author_text = ''
        publisher_text = ''
        if info_html:
            for item in info_html:
                if "/search/" in item['href'] or "/author/" in item['href']:
                    author_text += item.text + ' '
                # 提取出版社信息
                elif "book.douban.com/press/" in item['href']:
                    publisher_text = item.text
                    break

        # 提取书本图片信息
        img_html = soup.find('a', class_='nbg').find('img')
        match = re.search(r"https://.+\.jpg", str(img_html))
        image_url = match.group(0) if match else ''

        # 提取简介信息
        intro_info = soup.find('div', class_='intro')
        intro_text = intro_info.get_text(strip=True) if intro_info else ''
        return author_text, publisher_text, image_url, intro_text

    except Exception as ex:
        print(ex)
        return '', '', '', ''



if __name__ == '__main__':
    mysql = MysqlConnectorKeepAlive()
    LAST_WORK_ID = 1  # 上一次结束时的id
    work_ids = mysql.dql(f'select work_id from work where work_id >= {LAST_WORK_ID} order by work_id')

    # html = get_html(f'https://book.douban.com/subject/{LAST_WORK_ID}/')
    # author, publisher, image_url, intro = get_info(html)
    # print(author, publisher, image_url, intro)
    # exit()

    for work_id in tqdm(work_ids):
        html = get_html(f'https://book.douban.com/subject/{work_id}/')
        author, publisher, image_url, intro = get_info(html)

        author = re.sub(r'\s+', ' ', author.strip())    # 剔除多余空格
        author = author[:80]
        publisher = publisher[:80]
        image_url = image_url[:300]
        intro = intro[:300]     #

        author = author.replace("'", "''").replace("\\", "\\\\")   # 变换单引号与双斜杠
        publisher = publisher.replace("'", "''").replace("\\", "\\\\")   # 变换单引号与双斜杠
        image_url = image_url.replace("'", "''").replace("\\", "\\\\")   # 变换单引号与双斜杠
        intro = intro.replace("'", "''").replace("\\", "\\\\")   # 变换单引号与双斜杠

        # print(f'[{work_id}]', author, publisher, image_url, intro)
        if not mysql.dml(f"""
            update work
            set author = '{author}', publisher = '{publisher}', cover_link = '{image_url}', introduction = '{intro}'
            where work_id = {work_id}
        """):
            print(work_id)
            exit()

    mysql.close()