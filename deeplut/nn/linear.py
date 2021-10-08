class Linear(torch.nn.Linear):
    def __init__(self, in_features, out_features, binarize_input, k, minimal, binarized_calculation, device):
        real_in_features = in_features * (2 ** k)
        if minimal:
          tables_count = math.ceil(in_features/k)
          real_in_features = tables_count * (2 ** k)
        super(Linear, self).__init__(real_in_features, out_features)
        self.binarize_input = binarize_input
        self.k = k
        self.device = device
        self.input_mask = generate_input_mask(k, in_features, out_features, device, minimal)
        self.truth_table = generate_truth_table(k, in_features, out_features, device, minimal)
        self.binarized_calculation = binarized_calculation
        self.minimal = minimal

    def forward(self, input):
        if self.binarize_input:
            input.data = torch.sign(input.data)
        if not hasattr(self.weight, "org"):
            self.weight.org = self.weight.data.clone()
        self.weight.data = torch.sign(self.weight.org)
        expanded_input = input[:, self.input_mask]
        if self.minimal: 
          expanded_input=expanded_input.unsqueeze(-1)
        sum_input_table = expanded_input + self.truth_table
        if self.binarized_calculation:
          sum_input_table.data = torch.sign(sum_input_table.data)
        reduced_input = reduce_truth_table(self.k, sum_input_table, self.device)
        reduced_input = reduced_input.reshape(
            reduced_input.shape[0], self.out_features, -1
        )
        if self.binarized_calculation:
          reduced_input.data = torch.sign(reduced_input.data)
        reduced_input = reduced_input * self.weight

        if self.binarized_calculation:
          reduced_input.data = torch.sign(reduced_input.data)

        out = reduced_input.sum(dim=-1)
        if self.binarized_calculation:
           out.data = torch.sign(out.data)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        return out