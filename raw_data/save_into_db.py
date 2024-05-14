from tqdm import tqdm
from mysql_connector import mysql_connector_keep_alive
import pandas as pd

def read_data(file, names=None):
    return pd.read_csv(file, sep='::', engine='python', names=names)


'''将每一行作品数据插入work表'''
def load_book_data(mysql):
    data = read_data('Dou Ban Books Dataset/bookdata.dat')
    for row in tqdm(data.values):
        work_id, work_name, tags = row  # tags同时也会在其他函数插入tag表中
        sql = '''
              insert into work(work_id, work_name, tags, introduction, create_time, update_time)
              value({0},'{1}', '{2}', '这是从数据集导入的文学作品', now(), now())
              '''.format(work_id, work_name, tags)
        if not mysql.dml(sql):
            print('\t\t\t↑↑↑', work_id, work_name)


'''根据username插入user表中'''
def load_user_data(mysql):
    data = read_data('Dou Ban Books Dataset/ratingdata.dat')
    usernames = set()
    for row in data.values:   # 通过set去除重复username
        username, _, _ = row
        usernames.add(username)

    for username in tqdm(usernames):      # 将username插入到user表中
        sql = '''
              insert into user(username, password, introduction, update_time, create_time) 
              value('{0}', '123456', '这是从数据集导入的用户', now()), now()})
              '''.format(username)
        if not mysql.dml(sql):
            print('\t\t\t↑↑↑', username)


'''将评分记录插入review表'''
def load_rating_data_to_review(mysql):
    data = read_data('Dou Ban Books Dataset/ratingdata.dat')
    for row in tqdm(data.values):   # 将username变换为user_id 并插入评分表中
        username, book_id, rating = row
        sql = '''
              insert into review_user_work(user_id, work_id, rating, create_time, update_time)
              value((select distinct user.user_id from user where username = '{0}'), {1}, {2}, now(), now())
              '''.format(username, book_id, rating)
        if not mysql.dml(sql):
            print('\t\t\t↑↑↑', username, book_id, rating)


'''根据评分记录修改work表的两个rating字段'''
def load_rating_data_to_work(mysql):
    data = read_data('Dou Ban Books Dataset/ratingdata.dat')
    for row in tqdm(data.values):   # 将username变换为user_id 并插入评分表中
        username, book_id, rating = row
        sql = '''
              update work 
              set sum_rating = sum_rating + {0}, 
              sum_rating_user_number = sum_rating_user_number + 1,
              create_time = now(),
              update_time = now()
              where work_id = {1}
              '''.format(rating, book_id)
        if not mysql.dml(sql):
            print('\t\t\t↑↑↑', username, book_id, rating)

'''将作品标签插入tag和record_tag_work表中'''
def load_tags(mysql):
    data = read_data('Dou Ban Books Dataset/bookdata.dat')
    for row in tqdm(data.values):
        work_id, _, tags = row
        tag_list = str(tags).strip().split(' ')
        for tag_name in tag_list:
            sql = '''
                  select tag_id from tag where tag_name = '{0}'
                  '''.format(tag_name)
            result = mysql.dql(sql)
            if len(result) == 0:  # 这个tag没出现在tag表中
                # 先插入tag表
                sql = '''
                      insert into tag (tag_name, create_time, update_time)
                      value('{0}', now(), now())
                      '''.format(tag_name)
                if not mysql.dml(sql):
                    print('\t\t\t↑↑↑', tag_name)

                # 再寻找tag_id
                sql = '''
                      select tag_id from tag where tag_name = '{0}'
                      '''.format(tag_name)
                result = mysql.dql(sql)

            # 插入到tag_work_record中
            tag_id = result[0][0]
            sql = '''
                  insert into record_tag_work (tag_id, work_id, create_time, update_time)
                  value('{0}', '{1}', now(), now())
                  '''.format(tag_id, work_id)
            if not mysql.dml(sql):
                print('\t\t\t↑↑↑', tag_id, work_id)


'''去除没出现在bookdata中的所有"不存在"作品的评分记录'''
def delete_not_exist_books_rating(mysql):
    sql = '''
        delete from review_user_work ruw
        where not exists(
        select w.work_id from work w
        where w.work_id = ruw.work_id
    );
    '''
    mysql.dml(sql)


if __name__ == '__main__':
    mysql = mysql_connector_keep_alive.MysqlConnectorKeepAlive()
    load_book_data(mysql)
    load_user_data(mysql)
    load_rating_data_to_review(mysql)
    load_rating_data_to_work(mysql)
    delete_not_exist_books_rating(mysql)
    load_tags(mysql)
    mysql.close()
