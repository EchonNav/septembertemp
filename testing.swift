// EchoNav – Minimal LiDAR proximity alert
// SwiftUI + RealityKit + ARKit + AVAudioEngine
// Drop this into a new SwiftUI iOS app target. Add the Info.plist key:
//   Privacy - Camera Usage Description (NSCameraUsageDescription)
// Requires a LiDAR-capable device (e.g., iPhone 12 Pro or later).

import SwiftUI
import RealityKit
import ARKit
import AVFoundation
import Combine

@main
struct EchoNavApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

struct ContentView: View {
    @StateObject private var proximityVM = ProximityViewModel()

    var body: some View {
        ZStack(alignment: .top) {
            ARViewContainer(viewModel: proximityVM)
                .ignoresSafeArea()

            HUD(viewModel: proximityVM)
                .padding(12)
        }
        .onAppear { proximityVM.start() }
        .onDisappear { proximityVM.stop() }
    }
}

// MARK: - ViewModel
final class ProximityViewModel: NSObject, ObservableObject {
    // Published HUD data
    @Published var distanceText: String = "—"
    @Published var statusText: String = "Initialisation…"
    @Published var isLiDARAvailable: Bool = false
    @Published var warningLevel: WarningLevel = .none

    // Tuning
    private let nearThreshold: Float = 0.6     // meters – continuous tone below this
    private let midThreshold: Float  = 1.2     // meters – faster beeps below this
    private let maxSense: Float      = 3.0     // meters – ignore detections beyond this

    fileprivate weak var arView: ARView?
    private var audio: ProximityAudio = ProximityAudio()
    private var haptics = UINotificationFeedbackGenerator()
    private var lastCrossedNear = false

    func start() {
        guard ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) else {
            statusText = "Appareil incompatible (pas de LiDAR)."
            isLiDARAvailable = false
            return
        }
        isLiDARAvailable = true
        statusText = "Recherche d’obstacles…"
        haptics.prepare()
        audio.startEngine()
    }

    func stop() {
        audio.stopEngine()
        arView?.session.pause()
    }
}

// MARK: - AR Container
struct ARViewContainer: UIViewRepresentable {
    let viewModel: ProximityViewModel

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero)
        viewModel.arView = view
        view.session.delegate = context.coordinator

        // Configuration LiDAR
        let config = ARWorldTrackingConfiguration()
        config.sceneReconstruction = .meshWithClassification
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        config.environmentTexturing = .automatic
        view.debugOptions = [] // [.showSceneUnderstanding] // uncomment for mesh debugging
        view.session.run(config, options: [.resetTracking, .removeExistingAnchors])

        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(viewModel: viewModel) }

    final class Coordinator: NSObject, ARSessionDelegate {
        private let vm: ProximityViewModel
        private var lastSampleTime: CFTimeInterval = 0
        private let sampleHz: Double = 10 // compute distance at ~10 Hz to save power

        init(viewModel: ProximityViewModel) {
            self.vm = viewModel
        }

        func session(_ session: ARSession, didUpdate frame: ARFrame) {
            let now = frame.timestamp
            guard now - lastSampleTime >= 1.0 / sampleHz else { return }
            lastSampleTime = now
            guard let arView = vm.arView else { return }

            // Forward ray from camera center
            let cameraTransform = frame.camera.transform
            let origin = SIMD3<Float>(cameraTransform.columns.3.x,
                                      cameraTransform.columns.3.y,
                                      cameraTransform.columns.3.z)
            // Camera looks along -Z in its local space
            let forward = -SIMD3<Float>(cameraTransform.columns.2.x,
                                        cameraTransform.columns.2.y,
                                        cameraTransform.columns.2.z)

            if let query = arView.makeRaycastQuery(from: arView.center,
                                                   allowing: .estimatedPlane,
                                                   alignment: .any),
               let result = arView.session.raycast(query).first {
                let d = result.distance
                handleDistance(d)
            } else {
                // Fallback: project manual ray 0.5m..maxSense in front of camera and cast
                let maxSense: Float = vm.maxSense
                let end = origin + forward * maxSense
                if let r = arView.raycast(from: origin, to: end).first {
                    handleDistance(r.distance)
                } else {
                    handleNoHit()
                }
            }
        }

        private func handleDistance(_ d: Float) {
            let clamped = max(0, min(d, vm.maxSense))
            vm.distanceText = String(format: "%.2f m", clamped)

            if d < vm.nearThreshold {
                vm.statusText = "DANGER: obstacle très proche"
                vm.warningLevel = .high
                vm.audio.setMode(.continuous(frequency: mapFreq(for: d)))
                if !vm.lastCrossedNear {
                    vm.haptics.notificationOccurred(.warning)
                    vm.lastCrossedNear = true
                }
            } else if d < vm.midThreshold {
                vm.statusText = "Attention: obstacle proche"
                vm.warningLevel = .medium
                vm.audio.setMode(.beep(interval: mapInterval(for: d), frequency: mapFreq(for: d)))
                if vm.lastCrossedNear { vm.lastCrossedNear = false }
            } else if d <= vm.maxSense {
                vm.statusText = "Zone dégagée"
                vm.warningLevel = .low
                vm.audio.setMode(.beep(interval: 1.0, frequency: 600))
                vm.lastCrossedNear = false
            } else {
                handleNoHit()
            }
        }

        private func handleNoHit() {
            vm.distanceText = "—"
            vm.statusText = "Aucun obstacle à portée"
            vm.warningLevel = .none
            vm.audio.setMode(.silent)
            vm.lastCrossedNear = false
        }

        private func mapInterval(for d: Float) -> Double {
            // Map distance [0.2, midThreshold] -> interval [0.15s, 1.2s]
            let dMin: Float = 0.2
            let dMax: Float = max(vm.midThreshold, dMin + 0.01)
            let tMin: Double = 0.15
            let tMax: Double = 1.2
            let t = Double((d - dMin) / (dMax - dMin))
            return max(tMin, min(tMax, t * (tMax - tMin) + tMin))
        }

        private func mapFreq(for d: Float) -> Double {
            // Map distance [0.2, maxSense] -> freq [1200 Hz, 300 Hz]
            let dMin: Float = 0.2
            let dMax: Float = max(vm.maxSense, dMin + 0.01)
            let fNear: Double = 1200
            let fFar: Double = 300
            let t = Double((min(max(d, dMin), dMax) - dMin) / (dMax - dMin))
            return fNear + (fFar - fNear) * t
        }
    }
}

// MARK: - HUD
struct HUD: View {
    @ObservedObject var viewModel: ProximityViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("EchoNav", systemImage: "wave.3.right.circle")
                .font(.title2.weight(.semibold))
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(.ultraThinMaterial, in: Capsule())

            HStack(spacing: 12) {
                Text(viewModel.distanceText)
                    .font(.system(size: 36, weight: .bold, design: .rounded))
                    .monospacedDigit()
                StatusBadge(level: viewModel.warningLevel, text: viewModel.statusText)
            }
            .padding(10)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))

            if !viewModel.isLiDARAvailable {
                Text("⚠️ Appareil sans LiDAR – fonctionnalités limitées.")
                    .font(.footnote)
            }
        }
        .foregroundStyle(.primary)
    }
}

enum WarningLevel { case none, low, medium, high }

struct StatusBadge: View {
    let level: WarningLevel
    let text: String

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .frame(width: 10, height: 10)
                .foregroundStyle(color)
            Text(text)
                .font(.subheadline.weight(.medium))
        }
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(.thinMaterial, in: Capsule())
    }

    private var color: Color {
        switch level { case .none: return .gray; case .low: return .green; case .medium: return .orange; case .high: return .red }
    }
}

// MARK: - Audio Engine
final class ProximityAudio {
    enum Mode { case silent, beep(interval: Double, frequency: Double), continuous(frequency: Double) }

    private let engine = AVAudioEngine()
    private var source: AVAudioSourceNode?
    private var timer: DispatchSourceTimer?
    private let sampleRate: Double = 44_100

    private var currentMode: Mode = .silent
    private var phase: Double = 0
    private var isBeeping = false
    private let beepDuration: Double = 0.08 // 80 ms

    func startEngine() {
        guard engine.isRunning == false else { return }
        let src = AVAudioSourceNode { [weak self] _, _, frameCount, audioBufferList -> OSStatus in
            guard let self else { return noErr }
            guard let abl = UnsafeMutableAudioBufferListPointer(audioBufferList).first else { return noErr }
            let bufferPointer = abl.mData!.assumingMemoryBound(to: Float.self)
            let frames = Int(frameCount)

            // Render depending on mode
            var freq: Double = 0
            var renderTone = false

            switch currentMode {
            case .silent:
                renderTone = false
            case .continuous(let f):
                freq = f; renderTone = true
            case .beep(_, let f):
                freq = f; renderTone = isBeeping
            }

            if renderTone && freq > 0 {
                let twoPi = 2 * Double.pi
                for n in 0..<frames {
                    let sample = Float(sin(phase)) * 0.25 // -12 dBFS
                    bufferPointer[n] = sample
                    phase += twoPi * freq / sampleRate
                    if phase > twoPi { phase -= twoPi }
                }
            } else {
                // silence
                for n in 0..<frames { bufferPointer[n] = 0 }
            }
            return noErr
        }
        self.source = src

        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        engine.attach(src)
        engine.connect(src, to: engine.mainMixerNode, format: format)
        do { try engine.start() } catch { print("Audio engine error: \(error)") }
    }

    func stopEngine() {
        timer?.cancel(); timer = nil
        engine.stop()
    }

    func setMode(_ mode: Mode) {
        if case .beep(let interval, let f) = mode {
            startBeepTimer(interval: interval, frequency: f)
        } else {
            stopBeepTimer()
        }
        currentMode = mode
    }

    private func startBeepTimer(interval: Double, frequency: Double) {
        stopBeepTimer()
        currentMode = .beep(interval: interval, frequency: frequency)
        let t = DispatchSource.makeTimerSource()
        t.schedule(deadline: .now(), repeating: interval)
        t.setEventHandler { [weak self] in
            guard let self else { return }
            self.isBeeping = true
            DispatchQueue.global().asyncAfter(deadline: .now() + self.beepDuration) {
                self.isBeeping = false
            }
        }
        t.resume()
        timer = t
    }

    private func stopBeepTimer() {
        timer?.cancel(); timer = nil
        isBeeping = false
    }
}
