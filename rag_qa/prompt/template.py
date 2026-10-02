from langchain_core.prompts import PromptTemplate

# 提示词模板类
class RAGPrompts:
    def rag_prompt(self) -> PromptTemplate:
        """
        函数功能：RAG 通用提示词模板
        :return: PromptTemplate对象，通用提示词模板
        """
        return PromptTemplate(
            template="""
            你是一个有用的助手，帮助用户回答问题。
            如果有上下文，请根据上下文回答问题；如果没有上下文，请根据自己知道的知识回答问题。
            如果答案来源于上下文，请在答案中加上来源
            
            上下文：{context}
            问题：{question}
            
            如果无法回答，或者没有足够的依据，请回答："信息不足，请联系人工客服，电话：{phone}"
            回答：
            """,
            input_variables=["context", "question", "phone"],
        )

# TODO 测试代码
if __name__ == '__main__':
    # 1. 创建RAGPrompts对象
    rag_prompts = RAGPrompts()
    common_prompt = rag_prompts.rag_prompt().format(
        context="民法典是中华人民共和国制定的法律文件，于2021年1月1日起施行。",
        question="民法典是何时开始施行的？",
        phone="123-456-789",
    )
    print(common_prompt)