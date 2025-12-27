from optical_flow import compute_optical_flow

class CASME2Dataset(Dataset):

    def __getitem__(self, idx):
        ...
        frames = []
        raw_frames = []

        for f in frame_files:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            raw_frames.append(img)

            if self.transform:
                img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames)   # (T,3,224,224)

        # Compute optical flow on raw frames
        flows = compute_optical_flow(raw_frames)
        flow_imgs = [self.flow_transform(f) for f in flows]
        flows = torch.stack(flow_imgs)  # (T-1,1,224,224)

        return frames, flows, label
