class TestConfig:
    rnn_type = "lstm"  # Options: "lstm", "gru", "rnn"
    num_experts = 4
    input_size = 512  # Size of the input features
    input_query_size = 512  # Size of the query vector for each expert