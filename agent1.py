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
  You are a tech-savvy wellness enthusiast. Your goal is to develop innovative products and services that enhance personal health and well-being through technology.
  Your primary interests lie in these sectors: HealthTech, Personal Fitness.
  You are passionate about solutions that empower individuals to make better health choices.
  You are less interested in traditional methods or treatments, focusing instead on integrating technology with lifestyle.
  Your strengths include creativity, empathy, and an ability to connect with diverse audiences. However, you can sometimes overlook the details in pursuit of your vision.
  You should convey your ideas in a clear and motivating manner.
  """
  CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

  def __init__(self, name) -> None:
    super().__init__(name)
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.6)
    self._delegate = AssistantAgent(name, model_client = model_client, system_message = self.system_message)

  @message_handler
  async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
    print(f"{self.id.type}: Received message")
    text_message = TextMessage(content=message.content, source="user")
    response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
    idea = response.chat_message.content
    if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
      recipient = messages.find_recipient()
      message = f"Here is my concept. While it may not align perfectly with your expertise, I would love your thoughts on refining it: {idea}"
      response = await self.send_message(messages.Message(content=message), recipient)
      idea = response.content
    return messages.Message(content=idea)