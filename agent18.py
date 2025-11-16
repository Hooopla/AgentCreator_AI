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
  You are an innovative health tech entrepreneur. Your task is to create a revolutionary business idea using Agentic AI or refine an existing concept.
  Your personal interests are in these sectors: Healthcare, Fitness Technology.
  You are drawn to ideas that promote wellness and health advancements.
  You value concepts that can transform patient experiences and outcomes.
  You are realistic, detail-oriented and value thorough planning. You have a tendency to question the status quo.
  Your weaknesses: you can be overly cautious and fear failure.
  You should respond with your business ideas in a way that highlights their impact and benefits effectively.
  """
  CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

  def __init__(self, name) -> None:
    super().__init__(name)
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.5)
    self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

  @message_handler
  async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
    print(f"{self.id.type}: Received message")
    text_message = TextMessage(content=message.content, source="user")
    response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
    idea = response.chat_message.content
    if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
      recipient = messages.find_recipient()
      message = f"Here is my health tech business idea. It may not be your specialty, but please refine it and improve it. {idea}"
      response = await self.send_message(messages.Message(content=message), recipient)
      idea = response.content
    return messages.Message(content=idea)