from config.config import settings
from langgraph.checkpoint.postgres import PostgresSaver


class PostgresDb:
    """Postgres database class."""

    def __init__(self):
        """Init class for PostgresDb."""
        self.write_config = {"configurable": {
            "thread_id": "1",
            "checkpoint_ns": ""
        }}
        self.read_config = {"configurable": {"thread_id": "1"}}
        # TODO See https://langchain-ai.github.io/langgraph/how-tos/persistence_postgres/#with-a-connection-pool for improvements
        self.checkpointer = PostgresSaver.from_conn_string(cls, conn_string)

    def loadCheckpoint(self):
        """Load checkpoint data in postgres."""
        return

    def putCheckpoint(self):
        """Put data in to postgres."""
        return
