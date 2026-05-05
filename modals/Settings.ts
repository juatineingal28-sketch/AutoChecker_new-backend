// backend/models/Settings.ts
import mongoose, { Document, Schema } from 'mongoose';

export interface ISettings extends Document {
  userId: string;
  autoDetect: boolean;
  scanTips: boolean;
  flagLow: boolean;
  updatedAt: Date;
}

const SettingsSchema = new Schema<ISettings>(
  {
    userId:      { type: String, required: true, unique: true, index: true },
    autoDetect:  { type: Boolean, default: true },
    scanTips:    { type: Boolean, default: true },
    flagLow:     { type: Boolean, default: false },
  },
  { timestamps: true },
);

export const Settings = mongoose.model<ISettings>('Settings', SettingsSchema);