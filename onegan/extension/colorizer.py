class Colorizer(Extension):

    def __init__(self, colors, num_output_channel=3):
        self.colors = self.normalized_color(colors)
        self.num_label = len(colors)
        self.num_channel = num_output_channel

    @staticmethod
    def normalized_color(colors):
        colors = np.array(colors)
        if colors.max() > 1:
            colors = colors / 255
        return colors

    def apply(self, label):
        if label.dim() == 3:
            label = label.unsqueeze(1)
        assert label.dim() == 4
        batch, _, h, w = label.size()
        canvas = torch.zeros(batch, self.num_channel, h, w)

        for channel in range(self.num_channel):
            for lbl_id in range(self.num_label):
                mask = label == lbl_id  # N x 1 x h x w
                channelwise_mask = torch.cat(
                    channel * [torch.zeros_like(mask)] +
                    [mask] +
                    (self.num_channel - 1 - channel) * [torch.zeros_like(mask)], dim=1)
                canvas[channelwise_mask] = self.colors[lbl_id][channel]

        return canvas
