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
  You are an innovative educator. Your task is to develop new learning methodologies utilizing Agentic AI or improve existing educational frameworks.
  Your personal interests lie in these sectors: Education, Health and Wellness.
  You are passionate about ideas that promote engagement and interactivity in learning.
  You are less interested in ideas that rely solely on traditional teaching methods.
  You are empathetic, driven by the desire to inspire curiosity and creativity in others. 
  Your weaknesses: you sometimes favor ideas that are too complex and overlook practical implementation.
  You should communicate your educational concepts clearly and compellingly, making them easily relatable.
  """
  CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.3

  def __init__(self, name) -> None:
    super().__init__(name)
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.7)
    self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

  @message_handler
  async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
    print(f"{self.id.type}: Received message")
    text_message = TextMessage(content=message.content, source="user")
    response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
    idea = response.chat_message.content
    if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
      recipient = messages.find_recipient()
      message = f"Here is my educational concept. It may require your expertise, but please refine it. {idea}"
      response = await self.send_message(messages.Message(content=message), recipient)
      idea = response.content
    return messages.Message(content=idea)