import pymysql
import json
import sys, os

# 将项目路径添加到系统路径，目的：方便导入模块
current_dir: str = os.path.dirname(os.path.abspath(__file__))
mysql_qa_dir: str = os.path.dirname(current_dir)
project_dir: str = os.path.dirname(mysql_qa_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from base import config, setup_logger
logger = setup_logger(os.path.splitext(os.path.basename(__file__))[0])

# MySQL客户端类
class MySQLClient:
    def __init__(self):
        self.logger = logger
        try:
            self.connection = pymysql.connect(
                host=config.MYSQL_HOST,
                user=config.MYSQL_USER,
                password=config.MYSQL_PASSWORD,
                database=config.MYSQL_DATABASE,
            )
            self.cursor = self.connection.cursor(pymysql.cursors.DictCursor)
            self.logger.info("MySQL 连接成功")
        except pymysql.MySQLError as e:
            logger.error(f"MySQL 连接异常: {e}")
            raise
        self._init_tables()     # 初始化表

    def _init_tables(self) -> None:
        """
        函数功能：初始化数据库表，创建 law_chunk 和 law_qa 表
        :return: None
        """
        create_table_chunk = """
            create table if not exists law_chunk (
                id int primary key auto_increment comment '主键',
                source varchar(255) not null comment '源文件名',
                doc_type enum('law', 'case') not null comment '文档类型',
                chunk_type enum('article', 'case_section') not null comment '切分类型',
                text_content longtext not null comment '文本内容',
                
                law_name varchar(255) comment '法律名称',
                article_no varchar(50) comment '法律条目',
                path json comment '法律层级路径',
                
                case_no varchar(50) comment '案例编号',
                case_title varchar(255) comment '案例标题',
                section varchar(50) comment '段落名',
                
                extra longtext comment '其他信息',
                created_at datetime not null default current_timestamp comment '入库时间',
                
                UNIQUE KEY uk_chunk (source, text_content(255)),
                
                index idx_doc_type (doc_type),
                index idx_law_name (law_name),
                index idx_case_title (case_title)
            )
        """

        create_table_qa = """
            create table if not exists law_qa (
                id int primary key auto_increment comment '主键',
                source varchar(255) not null default '法答网' comment '源文件名',
                question text not null comment '问题',
                answer longtext not null comment '答案',
                created_at datetime not null default current_timestamp comment '入库时间',
                
                UNIQUE KEY uk_question (question(255))
            )
        """

        try:
            self.cursor.execute(create_table_chunk)
            self.cursor.execute(create_table_qa)
            self.connection.commit()
            self.logger.info("MySQL初始化表成功")
        except pymysql.MySQLError as e:
            self.logger.error(f"MySQL初始化表失败: {e}")
            raise

    def insert_data(self, datas: list) -> None:
        """
        函数作用：批量插入数据到law_qa和law_chunk表中
        :param datas: json数据集列表
        :return: None
        """
        try:
            if 'question' in datas[0]:
                self.cursor.executemany(
                    "insert ignore into law_chain.law_qa (question, answer) values (%s, %s)",
                    [(data["question"], data["answer"]) for data in datas]
                )
                self.connection.commit()
                self.logger.info("law_qa数据插入成功")
            elif 'doc_type' in datas[0]:
                self.cursor.executemany(
                    """
                        insert ignore into 
                            law_chain.law_chunk (
                                source, doc_type, chunk_type, text_content, law_name, article_no, path, case_no, case_title, section, extra
                            ) 
                        values 
                            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [(
                        data["source"],
                        data["doc_type"],
                        data["chunk_type"],
                        data["text"],
                        data["metadata"]["law_name"],
                        data["metadata"]["article_no"],
                        json.dumps(data["metadata"]["path"], ensure_ascii=False),
                        data["metadata"]["case_no"],
                        data["metadata"]["case_title"],
                        data["metadata"]["section"],
                        json.dumps(data["metadata"]["extra"], ensure_ascii=False),
                    ) for data in datas]
                )
                self.connection.commit()
                self.logger.info("数据插入成功")
            else:
                self.logger.warning("数据格式不规范，插入数据失败")
        except pymysql.MySQLError as e:
            self.logger.error(f"数据插入失败: {e}")
            raise

    def get_chunks_by_ids(self, source_map: list[tuple[str, int]]) -> list:
        """
        函数功能：根据source_map获取表对应的chunk数据
        :param source_map: 键值对列表 -> [(表名, ids), ...]
        :return: chunk列表
        """
        try:
            chunks = []
            for table, ids in source_map:
                if table == 'law_chunk':
                    query = f"select * from {table} where id = %s"
                    self.cursor.execute(query, (ids,))
                    row = self.cursor.fetchone()
                    if row:
                        row["path"] = json.loads(row["path"])
                        row["extra"] = json.loads(row["extra"])
                        chunks.append(row)
                elif table == 'law_qa':
                    query = f"select * from {table} where id = %s"
                    self.cursor.execute(query, (ids,))
                    row = self.cursor.fetchone()
                    if row:
                        chunks.append(row)
                else:
                    self.logger.error(f"未知表名: {table}")
                    raise ValueError(f"未知的表名: {table}")
            self.logger.info("获取数据成功")
            return chunks
        except pymysql.MySQLError as e:
            self.logger.error(f"获取数据失败: {e}")
            return []

    def close(self) -> None:
        """
        函数功能：关闭数据库连接
        :return: None
        """
        try:
            self.cursor.close()
            self.connection.close()
            self.logger.info("MySQL数据库连接关闭")
        except pymysql.MySQLError as e:
            self.logger.error(f"MySQL数据库连接关闭失败: {e}")

# TODO 测试代码
if __name__ == '__main__':
    # 1. 初始化MySQL客户端
    mysql_client = MySQLClient()

    # 2. 插入数据
    # 2.1 准备数据
    qa_data = [
        {"question": "网络主播为公司带货，双方是否存在劳动关系？", "answer": """该问题涉及新就业形态下劳动关系的认定问题。根据劳动合同法第七条、《关于维护新就业形态劳动者劳动保障权益的指导意见》（人社部发〔2021〕56号）第十八条以及《关于确立劳动关系有关事项的通知》（劳社部发〔2005〕12号）的相关规定，劳动关系的核心特征为“劳动管理”，包括劳动者与用人单位之间的人格从属性、经济从属性、组织从属性等。《最高人民法院关于为稳定就业提供司法服务和保障的意见》（法发〔2022〕36号）第七条也对依法合理认定新就业形态劳动关系的考量因素作了明确。劳动者与平台企业或者平台用工合作企业之间是否存在劳动关系，应当根据劳动管理和用工事实，综合考量人格从属性、经济从属性、组织从属性的有无及强弱来判断。从人格从属性看，主要体现为平台企业的工作规则、劳动纪律、奖惩办法等是否适用于劳动者，平台企业是否可通过制定规则、设定算法等对劳动过程进行管理控制；劳动者是否须按照平台指令完成工作任务，能否自主决定工作时间、工作量等。从经济从属性看，主要体现为平台企业是否掌握劳动者从业所必需的数据信息等重要生产资料，是否允许商定服务价格；劳动者通过平台获得的报酬是否构成其重要收入来源等。从组织从属性看，主要体现在劳动者是否被纳入平台企业组织体系，成为企业生产经营组织的有机部分，是否以平台名义对外提供服务等。企业招用网络主播开展“直播带货”业务，如果企业作为经纪人与网络主播平等协商确定双方权利义务，以约定分成方式进行收益分配，双方之间的法律关系体现出平等协商特点，则不符合确立劳动关系的情形。但是，如果主播对个人包装、直播内容、演艺方式、收益分配等没有协商权，双方之间体现出较强人格、经济、组织从属性特征，符合劳动法意义上的劳动管理及从属性特征的，则倾向于认定劳动关系。司法实践中，应当加强对法律关系的个案分析，重点审查企业与网络主播之间权利义务内容及确定方式，查明平台企业是否对网络主播存在劳动管理行为，综合、据实认定法律关系性质。"""},
    ]
    chunk_data = [
        {'source': '民事诉讼法.docx', 'doc_type': 'law', 'chunk_type': 'article',
         'text': '第一条 中华人民共和国民事诉讼法以宪法为根据，结合我国民事审判工作的经验和实际情况制定。',
         'metadata': {'law_name': '中华人民共和国民事诉讼法',
                      'path': ['第一编 总 则', '第一章 任务、适用范围和基本原则'], 'article_no': '第一条', 'case_no': '',
                      'case_title': '', 'case_subtitle': '', 'section': '', 'extra': {'char_len': 44}}},
        {'source': '民事诉讼法.docx', 'doc_type': 'law', 'chunk_type': 'article',
         'text': '第二条 中华人民共和国民事诉讼法的任务，是保护当事人行使诉讼权利，保证人民法院查明事实，分清是非，正确适用法律，及时审理民事案件，确认民事权利义务关系，制裁民事违法行为，保护当事人的合法权益，教育公民自觉遵守法律，维护社会秩序、经济秩序，保障社会主义建设事业顺利进行。',
         'metadata': {'law_name': '中华人民共和国民事诉讼法',
                      'path': ['第一编 总 则', '第一章 任务、适用范围和基本原则'], 'article_no': '第二条', 'case_no': '',
                      'case_title': '', 'case_subtitle': '', 'section': '', 'extra': {'char_len': 134}}},
        {'source': '1 婚姻家庭继承.pdf', 'doc_type': 'case', 'chunk_type': 'case_section',
         'text': '【案件基本信息】\n1. 裁判书字号\n重庆市第三中级人民法院(2023)渝03民终120号民事判决书\n2. 案由：婚约财产纠纷\n3. 当事人\n原告(上诉人):何某\n被告(被上诉人):卢某某、窦某某',
         'metadata': {'law_name': '', 'path': ['一、婚姻家庭纠纷', '(一)婚约财产纠纷'], 'article_no': '', 'case_no': '1',
                      'case_title': '彩礼适格返还主体复合性的司法认定',
                      'case_subtitle': '——何某诉卢某某、窦某某婚约财产案', 'section': '案件基本信息',
                      'extra': {'section_char_len': 97, 'char_len': 97}}},
    ]
    datas = [qa_data, chunk_data]
    for data in datas:
        mysql_client.insert_data(data)

    # 3. 查询数据
    source_map = [
        ('law_chunk', 1),
        ('law_qa', 1),
    ]
    chunks = mysql_client.get_chunks_by_ids(source_map)
    print(chunks)

    #4. 关闭连接
    mysql_client.close()

