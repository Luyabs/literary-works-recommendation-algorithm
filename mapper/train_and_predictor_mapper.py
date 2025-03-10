from mysql_connector import mysql_connector


def select_rating_by_user_id(user_id):
    """
    根据用户id查询对哪些文学作品发表了评价，并获取评分
    :param user_id: 用户id
    :return: column0 = work_id, column1 = rating,
    """
    mysql = mysql_connector.MysqlConnector()
    results = mysql.dql(f"""
        select work_id, rating from review_user_work 
        where user_id = {user_id}
        order by rating desc
        """
    )
    return results

def select_user_having_ratings_greater_equal(min_rating_num):
    """
    查找至少发表过min_rating_num个评分的用户
    :param min_rating_num: 最小发表的评价数量
    :return: list(user_id)
    """
    mysql = mysql_connector.MysqlConnector()
    results = mysql.dql(f"""
            select user_id from review_user_work
            group by user_id 
            having count(review_id) >= {min_rating_num}
        """
    )
    return results

def select_introduction_by_seq_work_id(seq_id):
    """
    通过work_id查询文学作品的简介
    :param seq_id: 经Embedding后的文学作品id （行号）
    :return: column0 = introduction
    """
    mysql = mysql_connector.MysqlConnector()
    results = mysql.dql(f"""
        select introduction from work LIMIT {seq_id}, 1
        """
    )
    return results

def select_tag_text_by_seq_work_id(seq_id):
    """
    通过work_id查询文学作品的标签文本
    :param seq_id: 经Embedding后的文学作品id （行号）
    :return: column0 = tag_text
    """
    mysql = mysql_connector.MysqlConnector()
    results = mysql.dql(f"""
        select tags from work LIMIT {seq_id}, 1
        """
    )
    return results