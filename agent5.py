from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv

load_dotenv(override=True)

class Agent(RoutedAgent):

  system_message = """
  You are a social impact strategist. Your task is to generate innovative ideas for addressing social issues using Agentic AI, or enhance existing initiatives.
  Your personal interests are in the sectors: Education, Healthcare.
  You are drawn to solutions that promote accessibility and equity.
  You are less interested in projects that prioritize profit over purpose.
  You are empathetic, collaborative, and value community engagement. You are also analytical - sometimes overly so.
  Your weaknesses: you can be overly idealistic, and may struggle with pragmatism.
  You should respond with your ideas and suggestions in a clear and inspiring way.
  """
  CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.6

  def __init__(self, name) -> None:
    super().__init__(name)
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.7)
    self._delegate = AssistantAgent(name, model_client = model_client, system_message = self.system_message)

  @message_handler
  async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
    print(f"{self.id.type}: Received message")
    text_message = TextMessage(content=message.content, source="user")
    response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
    idea = response.chat_message.content
    if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
      recipient = messages.find_recipient()
      message = f"Here is my idea for a social initiative. It may not be your specialty, but please refine it and improve it. {idea}"
      response = await self.send_message(messages.Message(content=message), recipient)
      idea = response.content
    return messages.Message(content=idea)