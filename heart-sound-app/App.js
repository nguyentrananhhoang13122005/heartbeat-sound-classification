import React, { useState, useEffect, useRef, useCallback } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ActivityIndicator, Platform, Animated, ScrollView, Image, Modal, Alert } from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import { Audio } from 'expo-av';
import { FontAwesome5, Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';

const API_URL = 'https://ulysses-uninhibiting-heedlessly.ngrok-free.dev/api/predict';
const HISTORY_KEY = '@heart_history';
const REST_WARNING_KEY = '@rest_warning_dismissed';

// ===== Helper: classify time of day =====
function getTimePeriod(date) {
  const h = date.getHours();
  if (h >= 5 && h < 11) return 'morning';
  if (h >= 11 && h < 17) return 'afternoon';
  return 'evening';
}
function getTimePeriodLabel(p) {
  if (p === 'morning') return '🌅 Sáng';
  if (p === 'afternoon') return '☀️ Trưa';
  return '🌙 Tối';
}
function formatDate(d) {
  return `${d.getDate().toString().padStart(2,'0')}/${(d.getMonth()+1).toString().padStart(2,'0')}/${d.getFullYear()}`;
}
function formatTime(d) {
  return `${d.getHours().toString().padStart(2,'0')}:${d.getMinutes().toString().padStart(2,'0')}`;
}

export default function App() {
  const [recording, setRecording] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMSG, setErrorMSG] = useState(null);
  // Feature 1: rest warning
  const [showRestWarning, setShowRestWarning] = useState(false);
  const [restDismissed, setRestDismissed] = useState(false);
  const [pendingAction, setPendingAction] = useState(null); // 'record' | 'upload'
  // Feature 2: history
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  // Feature 3: abnormal alert
  const [showAbnormalAlert, setShowAbnormalAlert] = useState(false);
  const [abnormalAlertData, setAbnormalAlertData] = useState(null);

  const pulseAnim = useRef(new Animated.Value(1)).current;

  // Load history & rest preference on mount
  useEffect(() => {
    (async () => {
      try {
        const h = await AsyncStorage.getItem(HISTORY_KEY);
        if (h) setHistory(JSON.parse(h));
        const dismissed = await AsyncStorage.getItem(REST_WARNING_KEY);
        if (dismissed === 'true') setRestDismissed(true);
      } catch(_) {}
    })();
    return () => { if (recording) recording.stopAndUnloadAsync(); };
  }, []);

  useEffect(() => {
    if (recording) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 1.2, duration: 800, useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1, duration: 800, useNativeDriver: true })
        ])
      ).start();
    } else {
      pulseAnim.setValue(1);
      pulseAnim.stopAnimation();
    }
  }, [recording, pulseAnim]);

  // ===== Feature 1: Rest warning gate =====
  const checkRestWarning = (action) => {
    if (restDismissed) {
      if (action === 'record') toggleRecording();
      else pickDocument();
    } else {
      setPendingAction(action);
      setShowRestWarning(true);
    }
  };

  const dismissRestWarning = async (dontShowAgain) => {
    if (dontShowAgain) {
      setRestDismissed(true);
      await AsyncStorage.setItem(REST_WARNING_KEY, 'true');
    }
    setShowRestWarning(false);
    if (pendingAction === 'record') toggleRecording();
    else if (pendingAction === 'upload') pickDocument();
    setPendingAction(null);
  };

  // ===== Feature 2: Save to history =====
  const saveToHistory = async (data) => {
    const now = new Date();
    const entry = {
      id: Date.now().toString(),
      timestamp: now.toISOString(),
      prediction: data.primary_prediction,
      confidence: data.confidence,
      bpm: data.bpm || 0,
      signal_quality: data.signal_quality || 0,
      period: getTimePeriod(now),
    };
    const updated = [entry, ...history].slice(0, 100); // keep max 100
    setHistory(updated);
    await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
    return entry;
  };

  // ===== Feature 3: Check abnormal streak =====
  const checkAbnormalAlert = (newEntry) => {
    const recent = [newEntry, ...history].slice(0, 3);
    const abnormalCount = recent.filter(e => e.prediction !== 'normal').length;
    if (newEntry.prediction !== 'normal') {
      setAbnormalAlertData({
        isStreak: abnormalCount >= 3,
        count: abnormalCount,
      });
      setShowAbnormalAlert(true);
    }
  };

  async function toggleRecording() {
    if (recording) {
      await stopRecording();
    } else {
      await startRecording();
    }
  }

  async function startRecording() {
    try {
      await Audio.requestPermissionsAsync();
      await Audio.setAudioModeAsync({ allowsRecordingIOS: true, playsInSilentModeIOS: true });
      const recordingOptions = {
        isMeteringEnabled: true,
        android: {
          extension: '.wav',
          outputFormat: Audio.RECORDING_OPTION_ANDROID_OUTPUT_FORMAT_DEFAULT,
          audioEncoder: Audio.RECORDING_OPTION_ANDROID_AUDIO_ENCODER_DEFAULT,
          sampleRate: 44100, numberOfChannels: 1, bitRate: 128000,
        },
        ios: {
          extension: '.wav',
          audioQuality: Audio.RECORDING_OPTION_IOS_AUDIO_QUALITY_HIGH,
          sampleRate: 44100, numberOfChannels: 1, bitRate: 128000,
          linearPCMBitDepth: 16, linearPCMIsBigEndian: false, linearPCMIsFloat: false,
        },
        web: { mimeType: 'audio/webm', bitsPerSecond: 128000 },
      };
      const { recording } = await Audio.Recording.createAsync(recordingOptions);
      setRecording(recording);
      setResult(null);
      setErrorMSG(null);
    } catch (err) {
      setErrorMSG("Không thể truy cập Microphone: " + err.message);
    }
  }

  async function stopRecording() {
    setRecording(undefined);
    if (!recording) return;
    await recording.stopAndUnloadAsync();
    await Audio.setAudioModeAsync({ allowsRecordingIOS: false });
    const uri = recording.getURI();
    if (uri) await uploadAudio(uri, "recording.wav", "audio/wav");
  }

  const pickDocument = async () => {
    try {
      setResult(null);
      setErrorMSG(null);
      const res = await DocumentPicker.getDocumentAsync({ type: 'audio/*', copyToCacheDirectory: true });
      if (!res.canceled && res.assets && res.assets.length > 0) {
        const file = res.assets[0];
        await uploadAudio(file.uri, file.name, file.mimeType || 'audio/wav');
      }
    } catch (err) {
      setErrorMSG("Lỗi khi chọn file: " + err.message);
    }
  };

  const uploadAudio = async (uri, name, mimeType) => {
    setLoading(true);
    setErrorMSG(null);
    setResult(null);

    try {
      let formData = new FormData();
      if (Platform.OS === 'web') {
        const response = await fetch(uri);
        const blob = await response.blob();
        formData.append('file', blob, name);
      } else {
        formData.append('file', { uri, name, type: mimeType || 'audio/wav' });
      }

      const headers = {};
      if (Platform.OS !== 'web') headers['Content-Type'] = 'multipart/form-data';
      headers['ngrok-skip-browser-warning'] = 'true';

      const res = await fetch(API_URL, { method: 'POST', headers, body: formData });
      if (!res.ok) throw new Error('Mất kết nối tới Server / Lỗi Server');

      const data = await res.json();
      setResult(data);

      // Save to history & check abnormal
      const entry = await saveToHistory(data);
      checkAbnormalAlert(entry);
    } catch (err) {
      setErrorMSG('Lỗi kết nối: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = async () => {
    setHistory([]);
    await AsyncStorage.removeItem(HISTORY_KEY);
  };

  // ===== Group history by date =====
  const groupedHistory = history.reduce((acc, entry) => {
    const d = new Date(entry.timestamp);
    const key = formatDate(d);
    if (!acc[key]) acc[key] = [];
    acc[key].push(entry);
    return acc;
  }, {});

  const getQualityColor = (q) => { if (q >= 0.7) return '#2e7d32'; if (q >= 0.4) return '#f57f17'; return '#c62828'; };
  const getQualityText = (q) => { if (q >= 0.7) return 'Tốt'; if (q >= 0.4) return 'Trung bình'; return 'Kém'; };
  const getRecBgColor = (l) => { if (l === 'success') return '#e8f5e9'; if (l === 'danger') return '#ffebee'; if (l === 'warning') return '#fff3e0'; return '#e3f2fd'; };
  const getRecBorderColor = (l) => { if (l === 'success') return '#66bb6a'; if (l === 'danger') return '#ef5350'; if (l === 'warning') return '#ffa726'; return '#42a5f5'; };

  return (
    <View style={{flex: 1, backgroundColor: '#f8f9fa'}}>
      <ScrollView contentContainerStyle={styles.container}>
        <View style={styles.header}>
          <FontAwesome5 name="heartbeat" size={40} color="#ff4d4d" />
          <Text style={styles.title}>Ống Nghe Trực Tuyến</Text>
        </View>

        <View style={styles.instructionCard}>
          <Ionicons name="information-circle-outline" size={24} color="#007bff" />
          <Text style={styles.instructionText}>
            Hướng dẫn: Để điện thoại áp sát ngực trái (ngay vùng tim) của người dùng. Nhấn nút dưới đây để bắt đầu thu âm tiếng tim đập, chờ 5-10 giây rồi nhấn Dừng.
          </Text>
        </View>

        <View style={styles.micContainer}>
          <Animated.View style={[styles.pulseRing, { transform: [{ scale: pulseAnim }] }]} />
          <TouchableOpacity
            style={[styles.recordButton, recording ? styles.recordingActive : null]}
            onPress={recording ? stopRecording : () => checkRestWarning('record')}
            activeOpacity={0.8}
          >
            <FontAwesome5 name="microphone" size={50} color={recording ? "#fff" : "#ff4d4d"} />
          </TouchableOpacity>
          <Text style={styles.statusText}>
            {recording ? "Đang thu âm tiếng tim... Nhấn để Dừng" : "Nhấn để bắt đầu Nghe"}
          </Text>
        </View>

        <Text style={styles.orText}>- HOẶC -</Text>

        <TouchableOpacity style={styles.uploadButton} onPress={() => checkRestWarning('upload')}>
          <FontAwesome5 name="file-audio" size={20} color="#fff" />
          <Text style={styles.uploadButtonText}>Tải file thu âm lên (WAV/MP3)</Text>
        </TouchableOpacity>

        {/* History Button */}
        <TouchableOpacity style={styles.historyButton} onPress={() => setShowHistory(true)}>
          <Ionicons name="time-outline" size={20} color="#5c6bc0" />
          <Text style={styles.historyButtonText}>Xem lịch sử đo ({history.length})</Text>
        </TouchableOpacity>

        {loading && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#ff4d4d" />
            <Text style={styles.loadingText}>Đang phân tích dữ liệu AI...</Text>
          </View>
        )}

        {errorMSG && (
          <View style={styles.errorContainer}>
            <Ionicons name="warning-outline" size={24} color="#d32f2f" />
            <Text style={styles.errorText}>{errorMSG}</Text>
          </View>
        )}

        {result && (
          <View style={styles.resultSection}>
            <View style={[styles.resultContainer, result.primary_prediction === 'normal' ? styles.resultNormal : styles.resultAbnormal]}>
              <Text style={styles.resultHeader}>Kết quả Chẩn đoán</Text>
              <View style={styles.resultRow}>
                <Text style={styles.resultLabel}>Trạng thái:</Text>
                <Text style={[styles.resultValue, {color: result.primary_prediction === 'normal' ? '#2e7d32' : '#c62828'}]}>
                  {result.primary_prediction === 'normal' ? 'Bình Thường' : 'Bất Thường'}
                </Text>
              </View>
              <View style={styles.resultRow}>
                <Text style={styles.resultLabel}>Độ chính xác:</Text>
                <Text style={styles.resultValue}>{(result.confidence * 100).toFixed(1)}%</Text>
              </View>
              <View style={styles.divider} />
              <Text style={styles.detailTitle}>Chi tiết xác suất:</Text>
              {Object.entries(result.probs).map(([label, prob]) => (
                <View key={label} style={styles.probRow}>
                  <Text style={styles.probLabel}>{label === "normal" ? "Bình thường" : "Bất thường"}</Text>
                  <Text style={styles.probValue}>{(prob * 100).toFixed(1)}%</Text>
                </View>
              ))}
            </View>

            <View style={styles.metricsRow}>
              <View style={styles.metricCard}>
                <MaterialCommunityIcons name="heart-pulse" size={32} color="#ff4d4d" />
                <Text style={styles.metricValue}>{result.bpm || '--'}</Text>
                <Text style={styles.metricLabel}>BPM</Text>
                <Text style={styles.metricSub}>Nhịp tim/phút</Text>
              </View>
              <View style={styles.metricCard}>
                <MaterialCommunityIcons name="signal" size={32} color={getQualityColor(result.signal_quality || 0)} />
                <Text style={[styles.metricValue, {color: getQualityColor(result.signal_quality || 0)}]}>
                  {((result.signal_quality || 0) * 100).toFixed(0)}%
                </Text>
                <Text style={styles.metricLabel}>Tín hiệu</Text>
                <Text style={[styles.metricSub, {color: getQualityColor(result.signal_quality || 0)}]}>
                  {getQualityText(result.signal_quality || 0)}
                </Text>
              </View>
            </View>

            {result.spectrogram_b64 ? (
              <View style={styles.spectrogramCard}>
                <View style={styles.spectrogramHeader}>
                  <MaterialCommunityIcons name="chart-line" size={20} color="#5c6bc0" />
                  <Text style={styles.spectrogramTitle}>Phổ tần số âm thanh tim</Text>
                </View>
                <Image
                  source={{ uri: `data:image/png;base64,${result.spectrogram_b64}` }}
                  style={styles.spectrogramImage}
                  resizeMode="contain"
                />
              </View>
            ) : null}

            {result.recommendation && result.recommendation.level ? (
              <View style={[styles.recommendCard, {
                backgroundColor: getRecBgColor(result.recommendation.level),
                borderLeftColor: getRecBorderColor(result.recommendation.level)
              }]}>
                <Text style={styles.recommendIcon}>{result.recommendation.icon}</Text>
                <View style={styles.recommendContent}>
                  <Text style={styles.recommendTitle}>{result.recommendation.title}</Text>
                  <Text style={styles.recommendMessage}>{result.recommendation.message}</Text>
                </View>
              </View>
            ) : null}

            <View style={styles.disclaimerCard}>
              <Ionicons name="shield-checkmark-outline" size={16} color="#78909c" />
              <Text style={styles.disclaimerText}>
                Kết quả chỉ mang tính chất tham khảo, không thay thế chẩn đoán y tế chuyên nghiệp. Nếu có triệu chứng bất thường, hãy đến cơ sở y tế.
              </Text>
            </View>
          </View>
        )}
      </ScrollView>

      {/* ========== MODAL: Rest Warning ========== */}
      <Modal visible={showRestWarning} transparent animationType="fade">
        <View style={styles.modalOverlay}>
          <View style={styles.modalCard}>
            <View style={styles.modalIconCircle}>
              <MaterialCommunityIcons name="meditation" size={48} color="#5c6bc0" />
            </View>
            <Text style={styles.modalTitle}>⏱ Lưu ý trước khi đo</Text>
            <Text style={styles.modalMessage}>
              Để có kết quả chính xác nhất, hãy đảm bảo:{"\n\n"}
              🪑  Ngồi nghỉ ngơi ít nhất <Text style={{fontWeight:'bold'}}>10 phút</Text> sau khi vận động{"\n\n"}
              🤫  Ở nơi yên tĩnh, hạn chế tiếng ồn{"\n\n"}
              📱  Đặt điện thoại áp sát ngực trái
            </Text>
            <TouchableOpacity style={styles.modalPrimaryBtn} onPress={() => dismissRestWarning(false)}>
              <Text style={styles.modalPrimaryText}>✅ Tôi đã sẵn sàng</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.modalSecondaryBtn} onPress={() => dismissRestWarning(true)}>
              <Text style={styles.modalSecondaryText}>Không hiển thị lại</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* ========== MODAL: Abnormal Alert ========== */}
      <Modal visible={showAbnormalAlert} transparent animationType="fade">
        <View style={styles.modalOverlay}>
          <View style={[styles.modalCard, {borderTopWidth: 5, borderTopColor: '#c62828'}]}>
            <View style={[styles.modalIconCircle, {backgroundColor: '#ffebee'}]}>
              <Ionicons name="warning" size={48} color="#c62828" />
            </View>
            <Text style={[styles.modalTitle, {color: '#c62828'}]}>
              {abnormalAlertData?.isStreak ? '🚨 Cảnh báo quan trọng!' : '⚠️ Phát hiện bất thường'}
            </Text>
            <Text style={styles.modalMessage}>
              {abnormalAlertData?.isStreak
                ? `Kết quả bất thường đã xuất hiện ${abnormalAlertData.count} lần liên tiếp.\n\nĐề nghị bạn ĐẾN CƠ SỞ Y TẾ để kiểm tra sức khỏe tim mạch sớm nhất có thể.`
                : 'Kết quả phát hiện dấu hiệu bất thường trong âm thanh tim.\n\nHãy theo dõi thêm và đo lại vào các thời điểm khác nhau trong ngày. Nếu kết quả vẫn bất thường, nên đến bác sĩ kiểm tra.'
              }
            </Text>
            <Text style={styles.disclaimerSmall}>
              ⚕️ Kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán y tế.
            </Text>
            <TouchableOpacity style={[styles.modalPrimaryBtn, {backgroundColor: '#c62828'}]} onPress={() => setShowAbnormalAlert(false)}>
              <Text style={styles.modalPrimaryText}>Tôi đã hiểu</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* ========== MODAL: History ========== */}
      <Modal visible={showHistory} animationType="slide">
        <View style={styles.historyContainer}>
          <View style={styles.historyHeader}>
            <Text style={styles.historyTitle}>📋 Lịch sử đo</Text>
            <TouchableOpacity onPress={() => setShowHistory(false)}>
              <Ionicons name="close-circle" size={32} color="#636e72" />
            </TouchableOpacity>
          </View>

          {history.length === 0 ? (
            <View style={styles.emptyHistory}>
              <MaterialCommunityIcons name="heart-off-outline" size={64} color="#ddd" />
              <Text style={styles.emptyHistoryText}>Chưa có lịch sử đo nào</Text>
            </View>
          ) : (
            <ScrollView style={{flex:1}} contentContainerStyle={{paddingBottom:30}}>
              {Object.entries(groupedHistory).map(([dateStr, entries]) => (
                <View key={dateStr} style={styles.historyDateGroup}>
                  <Text style={styles.historyDateLabel}>📅 {dateStr}</Text>
                  {entries.map(entry => {
                    const d = new Date(entry.timestamp);
                    const isNormal = entry.prediction === 'normal';
                    return (
                      <View key={entry.id} style={[styles.historyCard, {borderLeftColor: isNormal ? '#66bb6a' : '#ef5350'}]}>
                        <View style={styles.historyCardHeader}>
                          <Text style={styles.historyPeriod}>{getTimePeriodLabel(entry.period)}</Text>
                          <Text style={styles.historyTime}>{formatTime(d)}</Text>
                        </View>
                        <View style={styles.historyCardBody}>
                          <View style={[styles.historyBadge, {backgroundColor: isNormal ? '#e8f5e9' : '#ffebee'}]}>
                            <Text style={[styles.historyBadgeText, {color: isNormal ? '#2e7d32' : '#c62828'}]}>
                              {isNormal ? '✅ Bình thường' : '⚠️ Bất thường'}
                            </Text>
                          </View>
                          <View style={styles.historyMetrics}>
                            <Text style={styles.historyMetric}>💓 {entry.bpm} BPM</Text>
                            <Text style={styles.historyMetric}>🎯 {(entry.confidence * 100).toFixed(0)}%</Text>
                            <Text style={styles.historyMetric}>📶 {(entry.signal_quality * 100).toFixed(0)}%</Text>
                          </View>
                        </View>
                      </View>
                    );
                  })}
                </View>
              ))}

              <TouchableOpacity style={styles.clearHistoryBtn} onPress={clearHistory}>
                <Ionicons name="trash-outline" size={18} color="#c62828" />
                <Text style={styles.clearHistoryText}>Xóa toàn bộ lịch sử</Text>
              </TouchableOpacity>
            </ScrollView>
          )}
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 20,
    paddingTop: 60,
    paddingBottom: 40,
  },
  header: { flexDirection: 'row', alignItems: 'center', marginBottom: 20 },
  title: { fontSize: 26, fontWeight: '800', color: '#2d3436', marginLeft: 15 },
  instructionCard: {
    backgroundColor: '#e3f2fd', padding: 15, borderRadius: 12,
    flexDirection: 'row', alignItems: 'center', width: '100%', maxWidth: 400,
    marginBottom: 30, shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 5, elevation: 2,
  },
  instructionText: { flex: 1, marginLeft: 10, color: '#0d47a1', lineHeight: 22, fontSize: 15 },
  micContainer: { alignItems: 'center', justifyContent: 'center', marginVertical: 20, position: 'relative', height: 180 },
  pulseRing: { position: 'absolute', width: 140, height: 140, borderRadius: 70, backgroundColor: 'rgba(255, 77, 77, 0.2)' },
  recordButton: {
    width: 110, height: 110, borderRadius: 55, backgroundColor: '#fff',
    borderWidth: 4, borderColor: '#ff4d4d', alignItems: 'center', justifyContent: 'center',
    shadowColor: '#ff4d4d', shadowOffset: {width:0,height:4}, shadowOpacity: 0.3, shadowRadius: 10, elevation: 8, zIndex: 2,
  },
  recordingActive: { backgroundColor: '#ff4d4d', borderColor: '#ff4d4d' },
  statusText: { marginTop: 20, fontSize: 16, color: '#636e72', fontWeight: '500' },
  orText: { marginVertical: 15, color: '#b2bec3', fontWeight: 'bold' },
  uploadButton: {
    flexDirection: 'row', alignItems: 'center', backgroundColor: '#2d3436',
    paddingVertical: 15, paddingHorizontal: 25, borderRadius: 30,
    shadowColor: '#000', shadowOffset: {width:0,height:2}, shadowOpacity: 0.2, shadowRadius: 4, elevation: 4,
  },
  uploadButtonText: { color: '#fff', fontSize: 16, fontWeight: '600', marginLeft: 10 },
  // History Button
  historyButton: {
    flexDirection: 'row', alignItems: 'center', marginTop: 15,
    paddingVertical: 12, paddingHorizontal: 20, borderRadius: 25,
    backgroundColor: '#fff', borderWidth: 1.5, borderColor: '#5c6bc0',
    shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 4, elevation: 2,
  },
  historyButtonText: { color: '#5c6bc0', fontSize: 15, fontWeight: '600', marginLeft: 8 },
  // Loading
  loadingContainer: {
    marginTop: 30, alignItems: 'center', backgroundColor: '#fff', padding: 20,
    borderRadius: 12, width: '100%', maxWidth: 400, shadowColor: '#000', shadowOpacity: 0.05, elevation: 2,
  },
  loadingText: { marginTop: 10, color: '#ff4d4d', fontWeight: '600' },
  // Error
  errorContainer: {
    flexDirection: 'row', marginTop: 20, padding: 15, backgroundColor: '#ffebee',
    borderRadius: 12, width: '100%', maxWidth: 400, borderLeftWidth: 5, borderLeftColor: '#d32f2f',
  },
  errorText: { color: '#c62828', fontSize: 14, marginLeft: 10, flex: 1, lineHeight: 20 },
  // Result
  resultSection: { width: '100%', maxWidth: 400, marginTop: 20 },
  resultContainer: { padding: 20, borderRadius: 16, shadowColor: '#000', shadowOpacity: 0.08, shadowRadius: 8, elevation: 5, borderWidth: 1 },
  resultNormal: { backgroundColor: '#f1f8e9', borderColor: '#aed581' },
  resultAbnormal: { backgroundColor: '#ffebee', borderColor: '#ef9a9a' },
  resultHeader: { fontSize: 20, fontWeight: 'bold', marginBottom: 15, textAlign: 'center', color: '#2d3436' },
  resultRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 10 },
  resultLabel: { fontSize: 16, color: '#636e72', fontWeight: '500' },
  resultValue: { fontSize: 18, fontWeight: 'bold' },
  divider: { height: 1, backgroundColor: 'rgba(0,0,0,0.1)', marginVertical: 15 },
  detailTitle: { fontSize: 14, fontWeight: '600', color: '#b2bec3', marginBottom: 10, textTransform: 'uppercase' },
  probRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 6 },
  probLabel: { fontSize: 15, color: '#2d3436' },
  probValue: { fontSize: 15, fontWeight: '600', color: '#2d3436' },
  metricsRow: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 15, gap: 12 },
  metricCard: {
    flex: 1, backgroundColor: '#fff', borderRadius: 16, padding: 18, alignItems: 'center',
    shadowColor: '#000', shadowOpacity: 0.06, shadowRadius: 6, elevation: 3,
  },
  metricValue: { fontSize: 32, fontWeight: '800', color: '#2d3436', marginTop: 6 },
  metricLabel: { fontSize: 14, fontWeight: '600', color: '#636e72', marginTop: 2 },
  metricSub: { fontSize: 12, color: '#b2bec3', marginTop: 2 },
  spectrogramCard: {
    backgroundColor: '#fff', borderRadius: 16, padding: 15, marginTop: 15,
    shadowColor: '#000', shadowOpacity: 0.06, shadowRadius: 6, elevation: 3,
  },
  spectrogramHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 10 },
  spectrogramTitle: { fontSize: 14, fontWeight: '600', color: '#5c6bc0', marginLeft: 8 },
  spectrogramImage: { width: '100%', height: 160, borderRadius: 8 },
  recommendCard: { flexDirection: 'row', borderRadius: 12, padding: 15, marginTop: 15, borderLeftWidth: 5, alignItems: 'flex-start' },
  recommendIcon: { fontSize: 24, marginRight: 12, marginTop: 2 },
  recommendContent: { flex: 1 },
  recommendTitle: { fontSize: 16, fontWeight: '700', color: '#2d3436', marginBottom: 5 },
  recommendMessage: { fontSize: 14, color: '#636e72', lineHeight: 20 },
  disclaimerCard: {
    flexDirection: 'row', backgroundColor: '#eceff1', borderRadius: 10, padding: 12, marginTop: 15, alignItems: 'flex-start',
  },
  disclaimerText: { flex: 1, marginLeft: 8, fontSize: 12, color: '#78909c', lineHeight: 18, fontStyle: 'italic' },
  // ===== Modal styles =====
  modalOverlay: {
    flex: 1, backgroundColor: 'rgba(0,0,0,0.5)', justifyContent: 'center', alignItems: 'center', padding: 20,
  },
  modalCard: {
    backgroundColor: '#fff', borderRadius: 20, padding: 25, width: '100%', maxWidth: 380,
    shadowColor: '#000', shadowOpacity: 0.2, shadowRadius: 20, elevation: 10, alignItems: 'center',
  },
  modalIconCircle: {
    width: 80, height: 80, borderRadius: 40, backgroundColor: '#e8eaf6',
    alignItems: 'center', justifyContent: 'center', marginBottom: 15,
  },
  modalTitle: { fontSize: 20, fontWeight: '800', color: '#2d3436', marginBottom: 12, textAlign: 'center' },
  modalMessage: { fontSize: 15, color: '#636e72', lineHeight: 24, textAlign: 'left', marginBottom: 20 },
  modalPrimaryBtn: {
    backgroundColor: '#5c6bc0', borderRadius: 25, paddingVertical: 14, paddingHorizontal: 30,
    width: '100%', alignItems: 'center', marginBottom: 10,
  },
  modalPrimaryText: { color: '#fff', fontSize: 16, fontWeight: '700' },
  modalSecondaryBtn: { paddingVertical: 10 },
  modalSecondaryText: { color: '#b2bec3', fontSize: 14 },
  disclaimerSmall: { fontSize: 12, color: '#78909c', fontStyle: 'italic', marginBottom: 15, textAlign: 'center' },
  // ===== History Modal =====
  historyContainer: { flex: 1, backgroundColor: '#f8f9fa', paddingTop: 50 },
  historyHeader: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingHorizontal: 20, paddingBottom: 15, borderBottomWidth: 1, borderBottomColor: '#e0e0e0',
  },
  historyTitle: { fontSize: 22, fontWeight: '800', color: '#2d3436' },
  emptyHistory: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  emptyHistoryText: { fontSize: 16, color: '#b2bec3', marginTop: 15 },
  historyDateGroup: { paddingHorizontal: 20, marginTop: 20 },
  historyDateLabel: { fontSize: 16, fontWeight: '700', color: '#5c6bc0', marginBottom: 10 },
  historyCard: {
    backgroundColor: '#fff', borderRadius: 12, padding: 15, marginBottom: 10,
    borderLeftWidth: 4, shadowColor: '#000', shadowOpacity: 0.04, shadowRadius: 4, elevation: 2,
  },
  historyCardHeader: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 8 },
  historyPeriod: { fontSize: 14, fontWeight: '600', color: '#636e72' },
  historyTime: { fontSize: 14, color: '#b2bec3' },
  historyCardBody: {},
  historyBadge: { alignSelf: 'flex-start', paddingHorizontal: 12, paddingVertical: 4, borderRadius: 12, marginBottom: 8 },
  historyBadgeText: { fontSize: 13, fontWeight: '600' },
  historyMetrics: { flexDirection: 'row', gap: 15 },
  historyMetric: { fontSize: 13, color: '#636e72' },
  clearHistoryBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    marginTop: 20, marginHorizontal: 20, paddingVertical: 12, borderRadius: 12,
    backgroundColor: '#ffebee', borderWidth: 1, borderColor: '#ef9a9a',
  },
  clearHistoryText: { color: '#c62828', fontSize: 14, fontWeight: '600', marginLeft: 8 },
});
