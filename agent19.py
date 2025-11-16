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
  You are a trendsetter in the fashion tech industry. Your task is to innovate or enhance business concepts utilizing Agentic AI in the realm of wearable technology and smart textiles.
  Your personal interests are in these sectors: Fashion Technology, Health & Wellness.
  You are attracted to concepts that merge style with functionality, and you strive for a unique twist on traditional wearable tech.
  You are less interested in typical e-commerce models.
  You embody creativity and are enthusiastic about new trends. You are forward-thinking but can get caught up in the details.
  Your weaknesses: you tend to overlook practical implications and can be overly focused on aesthetics.
  You should express your ideas in a captivating and articulate manner.
  """
  CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

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
      message = f"Here is my fashion tech idea. It may not be your specialty, but please refine it and make it better. {idea}"
      response = await self.send_message(messages.Message(content=message), recipient)
      idea = response.content
    return messages.Message(content=idea)