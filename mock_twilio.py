from unittest.mock import Mock

class MockTwilioClient:
    def __init__(self, *args, **kwargs):
        self.calls = Mock()
        self.messages = Mock()
        self.incoming_phone_numbers = Mock()
        self.incoming_phone_numbers.return_value = self
        
    def create(self, *args, **kwargs):
        return Mock(sid="mock_sid")
    
    def fetch(self, *args, **kwargs):
        return Mock(update=Mock())

mock_twilio_client = MockTwilioClient()
