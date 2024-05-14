from mysql_connector import mysql_connector


def select_seq_user_id():
    """
    查询 user_id 与其所在的行号(从0开始）
    :return: column0 = user_id, column1 = seq_id, seq_id为连续的自然数序列
    """
    mysql = mysql_connector.MysqlConnector()
    results = mysql.dql(
        'select user_id, ROW_NUMBER() over (ORDER BY user_id) - 1 as seq_id from user;'
    )
    return results


def select_seq_book_id():
    """
    查询 work_id 与其所在的行号(从0开始）
    :return: column0 = work_id, column1 = seq_id, seq_id为连续的自然数序列
    """
    mysql = mysql_connector.MysqlConnector()
    results = mysql.dql(
        'select work_id, ROW_NUMBER() over (ORDER BY work_id) - 1 as seq_id from work;'
    )
    return results


def select_seq_book_tag_id():
    """
    查询 tag_id 与其所在的行号(从0开始）
    :return: column0 = tag_id, column1 = seq_id, seq_id为连续的自然数序列
    """
    mysql = mysql_connector.MysqlConnector()
    results = mysql.dql(
        'select tag_id, ROW_NUMBER() over (ORDER BY tag_id) - 1 as seq_id from tag;'
    )
    return results


def select_tag_id_by_book_id(book_id):
    """
    通过book_id 查询 该book的所有tag_id
    :param book_id: 文学作品id
    :return: list(tag_id)
    """
    mysql = mysql_connector.MysqlConnector()
    results = mysql.dql(
        f"select tag_id from record_tag_work where work_id = '{book_id}'"
    )
    return results

def select_count_tags():
    """
    返回总共有多少标签
    :return: list(count(tag_id))
    """
    mysql = mysql_connector.MysqlConnector()
    count = mysql.dql(
        f"select count(*) from tag"
    )
    return count