import json
import os

import pymysql
# import yaml


'''建立数据库连接(短连接, 每次执行完语句后自动close)'''
class BaseMysqlConnector:
    def __init__(self) -> None:
        self.db = None
        self.cursor = None


    '''select语句封装'''
    def dql(self, sql):
        pass

    '''insert/update/delete语句封装'''
    def dml(self, sql):
        pass


    '''建立数据库连接'''
    def open(self):
        db_config = self.__read_mysql_config__()['mysql']
        try:
            self.db = pymysql.connect(host=db_config['host'], port=db_config['port'],
                                      user=db_config['user'], password=db_config['password'], db=db_config['database'])
            self.cursor = self.db.cursor()
        except Exception as ex:
            print('连接数据库失败:', ex.args[1])
            exit()


    '''关闭数据库连接'''
    def close(self):
        self.db.close()


    # '''读取yaml配置文件'''
    # def __read_mysql_config_yaml__(self,file_name='config.yaml'):
    #     current_dir = os.path.dirname(__file__)
    #     project_dir = os.path.abspath(os.path.join(current_dir, '../utils', '..'))
    #     file_dir = os.path.join(project_dir, file_name)
    #     with open(file_dir) as yaml_file:
    #         config = yaml.safe_load(yaml_file)
    #     return config


    '''读取json配置文件'''
    def __read_mysql_config_json__(self, file_name='db_config.json'):
        current_dir = os.path.dirname(__file__)
        project_dir = os.path.abspath(os.path.join(current_dir, '../utils', '..'))
        file_dir = os.path.join(project_dir, file_name)

        with open(file_dir, 'r') as f:
            # 预处理步骤
            content = []
            for line in f:
                # 移除所有注释（//和#开头）
                clean_line = line.split('//')[0].split('#')[0].strip()
                if clean_line:
                    # 替换单引号为双引号
                    content.append(clean_line.replace("'", '"'))

            # 将多行合并为合法JSON
            json_str = ''.join(content)

            # 如果外层没有大括号则自动包裹
            if not json_str.startswith('{'):
                json_str = '{' + json_str + '}'

            # 解析JSON
            config = json.loads(json_str)

        return config