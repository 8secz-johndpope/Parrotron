from torch import nn

class ParrotronLoss(nn.Module):
    def __init__(self, spec_criterion, asr_criterion):
        super(ParrotronLoss, self).__init__()

        self.spec_criterion = spec_criterion
        self.asr_criterion = asr_criterion

    def forward(self, mel_outputs_postnet, mel_outputs, txt_outputs, targets, tts_targets):        
        targets.requires_grad = False
        tts_targets.requires_grad = False
        
        mel_loss = self.spec_criterion(mel_outputs_postnet, tts_targets) + self.spec_criterion(mel_outputs, tts_targets)
        asr_loss = self.asr_criterion(txt_outputs.contiguous().view(-1, txt_outputs.size(-1)), targets.contiguous().view(-1))
        
        return mel_loss + asr_loss


class ParrotronLossNoASR(nn.Module):
    def __init__(self, spec_criterion):
        super(ParrotronLossNoASR, self).__init__()
        self.spec_criterion = spec_criterion

    def forward(self, mel_outputs_postnet, mel_outputs, tts_targets):        
        tts_targets.requires_grad = False
        
        mel_loss = self.spec_criterion(mel_outputs_postnet, tts_targets) + self.spec_criterion(mel_outputs, tts_targets)
            
        return mel_loss