import torch
import torch.nn as nn

class ItemRS(nn.Module):
    def __init__(self, input_size, modalities, embedding_dim, avail_mode='none'):
        super(ItemRS, self).__init__()
        self.input_size = input_size
        self.modalities = modalities
        self.embedding_dim = embedding_dim
        self.n_modalities = len(modalities)

        self.encoder = nn.ModuleDict()
        for i in range(self.n_modalities):
            branch_name = modalities[i] + '_enc'
            mid = int(self.input_size[i] / 2)
            if self.embedding_dim < mid:
                mid = self.embedding_dim
            enc = nn.Sequential(
                nn.Linear(self.input_size[i], mid),
                nn.LeakyReLU(),
                nn.Linear(mid, self.embedding_dim),
                nn.LeakyReLU(),
            )
            self.encoder[branch_name] = enc

        self.out = nn.Sequential(
            nn.Linear(self.embedding_dim * self.n_modalities, self.embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(self.embedding_dim, 1),
        )
        self.sigmoid = nn.Sigmoid()

        # Availability-aware head (--irs_avail): lets sigm(y_i) see which modalities
        # the item actually has (indicator o_i, 1 = observed). 'none' keeps the original flow.
        if avail_mode not in ('none', 'concat', 'only'):
            raise ValueError("unknown avail_mode %r" % avail_mode)
        self.avail_mode = avail_mode
        if avail_mode == 'concat':      # content + availability flags
            self.out_avail = nn.Sequential(
                nn.Linear(self.embedding_dim * self.n_modalities + self.n_modalities, self.embedding_dim),
                nn.LeakyReLU(),
                nn.Linear(self.embedding_dim, 1),
            )
        elif avail_mode == 'only':      # availability flags alone
            self.out_avail = nn.Sequential(
                nn.Linear(self.n_modalities, 16),
                nn.LeakyReLU(),
                nn.Linear(16, 1),
            )

    def forward_avail(self, x, avail):
        """Item score from the availability-aware head (--irs_avail concat/only).

        x: list of per-modality features [B, d_m]; avail: [B, n_modalities] float, 1 = observed.
        Returns (y_i, sigm(y_i)) like forward().
        """
        if self.avail_mode == 'only':
            out = self.out_avail(avail)
        elif self.avail_mode == 'concat':
            encoded = [self.encoder[self.modalities[i] + '_enc'](x[i]) for i in range(self.n_modalities)]
            out = self.out_avail(torch.cat(encoded + [avail], 1))
        else:
            raise ValueError("forward_avail needs avail_mode 'concat' or 'only', got %r" % self.avail_mode)
        return out, self.sigmoid(out)

    def forward(self, x):
        encoded = []
        for i in range(self.n_modalities):
            branch_name = self.modalities[i] + '_enc'
            enc = self.encoder[branch_name](x[i])
            encoded.append(enc)
        encoded = torch.cat(encoded, 1)
        out = self.out(encoded)
        out_sigmoid = self.sigmoid(out)
        return out, out_sigmoid