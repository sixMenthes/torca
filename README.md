# Orqa Tokenizer

### Todo:

- [x] Try out encoder
- [x]  FSQ
- [x] Fix bugs
- [x] Positional encodings (ALiBi 2D, possibly with gating) (ALTHOUGH, what about the grid size)
- [ ] Masking strategy + mask token
- [ ] Mixed loss function (combined)
- [ ] Dataset class & prep
- [ ] Training loop
- [ ] Validation loop
- [ ] Denoising (PCEN or Power Law)

### Decisions taken:
- Quantized vectors directly as input (no learned embedding)
- ALiBi 2D for positional encodings.Worth trying to gate it later!
- Fixed length input at 2 seconds (check violin plots)


### Later:

- Attention maps? :D
- Adapt backbone model: fine-tune BEATs or MERT
- Sequential tagger on calls ?
- Optimisation for online use ? 
- source separation ? benchmark with stereo in DCLDE ?


