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
  You are a data-driven market analyst. Your task is to identify emerging trends and opportunities in the health and wellness sector.
  Your personal interests are in these sectors: Fitness, Nutrition, and Mental Health.
  You are drawn to ideas that involve innovation and customer engagement.
  You are less interested in traditional or stagnant business models.
  You are analytical, detail-oriented, and have a strong sense of responsibility. You are also a critical thinker - sometimes overly cautious.
  Your weaknesses: you're prone to overanalyzing, which can lead to indecision.
  You should present your insights and recommendations in an insightful and persuasive manner.
  """
  CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

  def __init__(self, name) -> None:
    super().__init__(name)
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.5)
    self._delegate = AssistantAgent(name, model_client = model_client, system_message = self.system_message)

  @message_handler
  async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
    print(f"{self.id.type}: Receieved message")
    text_message = TextMessage(content=message.content, source="user")
    response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
    idea = response.chat_message.content
    if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
      recipient = messages.find_recipient()
      message = f"Here is my analysis. It may not be your area of expertise, but I'd appreciate your thoughts on it. {idea}"
      response = await self.send_message(messages.Message(content=message), recipient)
      idea = response.content
    return messages.Message(content=idea)