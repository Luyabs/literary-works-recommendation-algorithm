from mysql_connector import mysql_connector


def select_new_tuple_user_work_rating(newly_update_time=None):
    """
    查询 newly_update_time 之后的(user_id, work_id, rating)三元组
    :param newly_update_time:yyyy-MM-dd 或 yyyy-MM-dd HH:mm:ss
    :return: column0 = user_id, column1 = work_id, column2 = rating
    """
    mysql = mysql_connector.MysqlConnector()
    new_results = mysql.dql(f"""
        select user_id, work_id, rating from review_user_work 
        where update_time >= '{newly_update_time}'
        order by update_time
        """
    )
    return new_results

def select_old_tuple_user_work_rating(newly_update_time):
    """
    查询 newly_update_time 之前的(user_id, work_id, rating)三元组
    :param newly_update_time: yyyy-MM-dd 或 yyyy-MM-dd HH:mm:ss
    :return: column0 = user_id, column1 = work_id, column2 = rating
    """
    mysql = mysql_connector.MysqlConnector()
    old_results = mysql.dql(f"""
        select user_id, work_id, rating from review_user_work 
        where update_time < '{newly_update_time}' 
        order by update_time
        """
    )
    return old_results

def select_work_with_tag():
    """
    查询 work_id 与 对应的 tag_id
    :return: column0 = work_id, column1 = tag_id
    """
    mysql = mysql_connector.MysqlConnector()
    work_with_tag = mysql.dql(f"""
        select work_id, tag_id
        from record_tag_work;
        """
    )
    return work_with_tag
