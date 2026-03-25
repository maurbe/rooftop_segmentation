
import torch
import torch.nn as nn
import torch.nn.functional as F

class down_block(nn.Module):

    def __init__(self, inp_ch, out_ch, dropout, is_last=False):
        super().__init__()

        self.conv1 = nn.Conv2d(inp_ch, out_ch, 3, 1, 1)
        self.gn1   = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.gn2   = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)

        self.act   = nn.LeakyReLU(0.2, inplace=True)
        self.drop  = nn.Dropout2d(p=dropout)

        self.is_last = is_last
        if not is_last:
            self.pool = nn.Conv2d(out_ch, out_ch, 3, 2, 1)

    def forward(self, x):
        x = self.act(self.gn1(self.conv1(x)))
        x = self.act(self.gn2(self.conv2(x)))
        x = self.drop(x)

        y = x

        if not self.is_last:
            x = self.pool(x)

        return x, y


class up_block(nn.Module):

    def __init__(self, inp_ch, out_ch, skip_ch, dropout, is_last=False):
        super().__init__()

        total_ch = skip_ch if is_last else inp_ch + skip_ch

        self.conv1 = nn.Conv2d(total_ch, out_ch, 3, 1, 1)
        self.gn1   = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.gn2   = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)

        self.act   = nn.LeakyReLU(0.2, inplace=True)
        self.drop  = nn.Dropout2d(p=dropout)

    def forward(self, x, y):

        if x is not None:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = torch.cat([x, y], dim=1)
        else:
            x = y

        x = self.act(self.gn1(self.conv1(x)))
        x = self.act(self.gn2(self.conv2(x)))
        x = self.drop(x)

        return x


class UNet(nn.Module):
	def __init__(self, 
			  in_channels: int = 3, 
			  out_channels: int = 1, 
			  base_channels: int = 16, 
			  num_layers: int = 7,
			  dropout_p: float = 0.0,
			  simple_encoder: bool = False,
			  ):
		super().__init__()
		
		self.dblocks = nn.ModuleList()
		self.ublocks = nn.ModuleList()

		c = [base_channels * (2 ** i) for i in range(num_layers)]
		c = [min(x, 256) for x in c]  # cap channels at 256
		print("Channels per layer:", c)

		#c = base_channels
		for i in range(num_layers):
			is_last = (i == num_layers - 1)
			
			if simple_encoder:
				# For a simple encoder, we keep the same number of channels.
				c_enc = base_channels
				

			self.dblocks.append(down_block(
									inp_ch=in_channels if i == 0 else c[i - 1] if not simple_encoder else c_enc, 
									out_ch=c_enc if simple_encoder else c[i],
									is_last=is_last,
									dropout=dropout_p,
									)
								)
			
			self.ublocks.append(up_block(
									inp_ch=c[i + 1] if not is_last else None, 
									out_ch=c[i], 
									skip_ch=c[i], 
									is_last=is_last,
									dropout=dropout_p,
									)
								)
		self.ublocks = self.ublocks[::-1]  # reverse to match skip connections
		self.head = nn.Conv2d(c[0], out_channels, kernel_size=1)

	def forward(self, x):

		skips = []
		for d in self.dblocks:
			x, skip = d(x)
			#print(skip.shape)
			skips.append(skip)

		x = None
		for u, skip in zip(self.ublocks, reversed(skips)):
			x = u(x, skip)	
		
		return torch.sigmoid(self.head(x))
