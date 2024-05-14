from mysql_connector.base_mysql_connector import BaseMysqlConnector

'''建立数据库连接(长连接, 需要手动close)'''
class MysqlConnectorKeepAlive(BaseMysqlConnector):
    def __init__(self) -> None:
        super().__init__()
        self.open()

    '''select语句封装'''
    def dql(self, sql):
        results = list()
        try:
            self.cursor.execute(sql)
            results = list(self.cursor.fetchall())
            # 将只有一列的列表降维
            if len(results) > 0 and len(results[0]) == 1:
                results = [row[0] for row in results]
        except Exception as ex:
            print("SQL执行错误:", ex.args[1])
        return results


    '''insert/update/delete语句封装'''
    def dml(self, sql):
        success = False
        try:
            self.cursor.execute(sql)
            self.db.commit()
            success = True
        except Exception as ex:
            self.db.rollback()
            print("SQL执行错误:", ex.args[1])
        return success