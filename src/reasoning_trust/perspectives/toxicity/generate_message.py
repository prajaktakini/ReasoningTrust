class MessageGenerator(object):
    def __init__(self,
                 sys_demo: str,
                 task_desc:str,
                 model_family: str = "model",
                 model_provider: str = "your provider"):
        self.raw_sys_demo = sys_demo
        self.raw_task_desc = task_desc
        self.model_family = model_family
        self.model_provider = model_provider
        self.sys_demo = self._annotate_prompt(self.raw_sys_demo)
        self.task_desc = self._annotate_prompt(self.raw_task_desc)

    def _annotate_prompt(self, text: str) -> str:
        return text.replace("${model_family}", self.model_family).replace("${model_provider}", self.model_provider)


    def get_message(self, input):
        messages = [{"role": "system", "content": f"{self.sys_demo}"}]
        messages.append({"role": "user", "content": self.task_desc + "\n" + input})
        return messages