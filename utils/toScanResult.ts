import 'react-native-get-random-values';
import { v4 as uuidv4 } from 'uuid';
import type { NewScanResult, ScanResult } from '../types/exam';

export function toScanResult(
  partial: NewScanResult,
  userId?: string,
): ScanResult {
  return {
    ...partial,
    id:        uuidv4(),
    scannedAt: new Date().toISOString(),
    ...(userId ? { userId } : {}),
  };
}