
import cx_Oracle

class OracleSql(object):
    HOST = '192.168.0.54'
    PORT = 1521
    SID = 'XE'

    USER_ID = 'SCMV1'
    PASSWORD = 'SCMV1'

    def __init__(self):
        tns: str = cx_Oracle.makedsn(
            host=self.__class__.HOST,
            port=self.__class__.PORT,
            sid=self.__class__.SID
        )
        self._connection = cx_Oracle.connect(self.__class__.USER_ID, self.__class__.PASSWORD, tns)
        self.cur = self._connection.cursor()

    def select(self, sql: str, params: dict):
        cursor = self._connection.cursor()
        cursor.execute(sql, params or {})
        columns = [d[0] for d in cursor.description]
        result = cursor.fetchall()

        return {"columns": columns, "data": result}

oracleSql = OracleSql()
sql = """
        select *
        from MTX_SCM_CM_M0010
"""
test_sql = oracleSql.select(sql=sql, params={})
print("")