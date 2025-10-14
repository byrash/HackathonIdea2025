import { Component, Input, Output, EventEmitter, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AnalysisResults, ApiService } from '../services/api.service';

@Component({
  selector: 'app-results',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="space-y-6">
      <!-- Verdict Card -->
      <div class="bg-white rounded-lg shadow-lg p-8">
        <div class="flex items-center justify-between mb-6">
          <h2 class="text-2xl font-bold text-gray-800">Analysis Results</h2>
          <button (click)="newAnalysis.emit()" class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">New Analysis</button>
        </div>

        <!-- Risk Score & Verdict -->
        <div class="grid md:grid-cols-2 gap-6 mb-6">
          <!-- Risk Score -->
          <div
            class="p-6 bg-gradient-to-br rounded-lg"
            [class.from-green-50]="results.overallRiskScore < 30"
            [class.to-green-100]="results.overallRiskScore < 30"
            [class.from-yellow-50]="results.overallRiskScore >= 30 && results.overallRiskScore < 60"
            [class.to-yellow-100]="results.overallRiskScore >= 30 && results.overallRiskScore < 60"
            [class.from-red-50]="results.overallRiskScore >= 60"
            [class.to-red-100]="results.overallRiskScore >= 60"
          >
            <h3 class="text-sm font-medium text-gray-600 mb-2">Fraud Risk Score</h3>
            <div class="flex items-end space-x-2">
              <span
                class="text-5xl font-bold"
                [class.text-green-700]="results.overallRiskScore < 30"
                [class.text-yellow-700]="results.overallRiskScore >= 30 && results.overallRiskScore < 60"
                [class.text-red-700]="results.overallRiskScore >= 60"
              >
                {{ results.overallRiskScore }}
              </span>
              <span class="text-2xl font-medium text-gray-600 mb-2">/100</span>
            </div>
            <div class="mt-3">
              <div class="w-full bg-white bg-opacity-50 rounded-full h-2">
                <div
                  class="h-2 rounded-full transition-all"
                  [class.bg-green-600]="results.overallRiskScore < 30"
                  [class.bg-yellow-600]="results.overallRiskScore >= 30 && results.overallRiskScore < 60"
                  [class.bg-red-600]="results.overallRiskScore >= 60"
                  [style.width.%]="results.overallRiskScore"
                ></div>
              </div>
            </div>
          </div>

          <!-- Verdict -->
          <div
            class="p-6 rounded-lg border-2"
            [class.border-green-500]="results.verdict === 'LEGITIMATE'"
            [class.bg-green-50]="results.verdict === 'LEGITIMATE'"
            [class.border-yellow-500]="results.verdict === 'SUSPICIOUS'"
            [class.bg-yellow-50]="results.verdict === 'SUSPICIOUS'"
            [class.border-red-500]="results.verdict === 'FRAUDULENT'"
            [class.bg-red-50]="results.verdict === 'FRAUDULENT'"
          >
            <h3 class="text-sm font-medium text-gray-600 mb-2">Verdict</h3>
            <div class="flex items-center space-x-3">
              <div class="text-4xl">
                <span *ngIf="results.verdict === 'LEGITIMATE'">✓</span>
                <span *ngIf="results.verdict === 'SUSPICIOUS'">⚠️</span>
                <span *ngIf="results.verdict === 'FRAUDULENT'">✕</span>
              </div>
              <div>
                <p
                  class="text-3xl font-bold"
                  [class.text-green-700]="results.verdict === 'LEGITIMATE'"
                  [class.text-yellow-700]="results.verdict === 'SUSPICIOUS'"
                  [class.text-red-700]="results.verdict === 'FRAUDULENT'"
                >
                  {{ results.verdict }}
                </p>
                <p class="text-sm text-gray-600 mt-1">
                  {{ getVerdictDescription(results.verdict) }}
                </p>
              </div>
            </div>
          </div>
        </div>

        <!-- Recommendations -->
        <div class="p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <h3 class="font-semibold text-gray-800 mb-3">Recommendations:</h3>
          <ul class="space-y-2">
            <li *ngFor="let rec of results.recommendations" class="flex items-start space-x-2">
              <span class="text-blue-600 mt-1">•</span>
              <span class="text-sm text-gray-700">{{ rec }}</span>
            </li>
          </ul>
        </div>
      </div>

      <!-- Extracted Data -->
      <div class="bg-white rounded-lg shadow-lg p-6">
        <h3 class="text-xl font-bold text-gray-800 mb-4">Extracted Check Data</h3>
        <div class="grid md:grid-cols-2 gap-4">
          <div class="p-3 bg-gray-50 rounded">
            <p class="text-xs text-gray-600">Check Number</p>
            <p class="font-medium">{{ results.extractedData.checkNumber || 'N/A' }}</p>
          </div>
          <div class="p-3 bg-gray-50 rounded">
            <p class="text-xs text-gray-600">Date</p>
            <p class="font-medium">{{ results.extractedData.date || 'N/A' }}</p>
          </div>
          <div class="p-3 bg-gray-50 rounded">
            <p class="text-xs text-gray-600">Payee</p>
            <p class="font-medium">{{ results.extractedData.payee || 'N/A' }}</p>
          </div>
          <div class="p-3 bg-gray-50 rounded">
            <p class="text-xs text-gray-600">Amount</p>
            <p class="font-medium">{{ formatAmount(results.extractedData.amountNumeric) }}</p>
          </div>
          <div class="p-3 bg-gray-50 rounded">
            <p class="text-xs text-gray-600">Routing Number</p>
            <p class="font-medium font-mono">{{ results.extractedData.routingNumber || 'N/A' }}</p>
          </div>
          <div class="p-3 bg-gray-50 rounded">
            <p class="text-xs text-gray-600">Account Number</p>
            <p class="font-medium font-mono">****{{ results.extractedData.accountNumber || 'N/A' }}</p>
          </div>
        </div>
      </div>

      <!-- Analysis Details -->
      <div class="bg-white rounded-lg shadow-lg p-6">
        <h3 class="text-xl font-bold text-gray-800 mb-4">Detailed Analysis</h3>

        <!-- Forensic Analysis -->
        <div class="mb-6">
          <h4 class="font-semibold text-gray-700 mb-2">🔬 Forensic Analysis</h4>
          <div class="space-y-2 text-sm">
            <p *ngFor="let summary of results.forensicAnalysis?.summary" class="text-gray-600">• {{ summary }}</p>
          </div>
          <p class="text-xs text-gray-500 mt-2">Risk Score: {{ results.forensicAnalysis?.overall_risk_score || 0 }}/100</p>
        </div>

        <!-- Security Features -->
        <div class="mb-6">
          <h4 class="font-semibold text-gray-700 mb-2">🔒 Security Features</h4>
          <div class="space-y-2 text-sm">
            <div class="flex items-center space-x-2">
              <span [class.text-green-600]="results.securityFeatures?.watermarkDetected" [class.text-red-600]="!results.securityFeatures?.watermarkDetected">
                {{ results.securityFeatures?.watermarkDetected ? '✓' : '✗' }}
              </span>
              <span class="text-gray-600">Watermark Detection</span>
            </div>
            <div class="flex items-center space-x-2">
              <span [class.text-green-600]="results.securityFeatures?.templateMatch?.matched" [class.text-red-600]="!results.securityFeatures?.templateMatch?.matched">
                {{ results.securityFeatures?.templateMatch?.matched ? '✓' : '✗' }}
              </span>
              <span class="text-gray-600">Template Match</span>
            </div>
            <div class="flex items-center space-x-2">
              <span [class.text-green-600]="results.securityFeatures?.securityPatterns?.patterns_detected" [class.text-red-600]="!results.securityFeatures?.securityPatterns?.patterns_detected">
                {{ results.securityFeatures?.securityPatterns?.patterns_detected ? '✓' : '✗' }}
              </span>
              <span class="text-gray-600">Security Patterns</span>
            </div>
          </div>
        </div>

        <!-- ML Prediction -->
        <div>
          <h4 class="font-semibold text-gray-700 mb-2">🤖 ML Fraud Detection</h4>
          <div class="text-sm text-gray-600">
            <p>Fraud Probability: {{ (results.mlPrediction?.fraud_probability * 100).toFixed(1) }}%</p>
            <p *ngIf="results.mlPrediction?.fraud_type">
              Detected Type: <span class="font-medium">{{ results.mlPrediction.fraud_type }}</span>
            </p>
            <p class="text-xs text-gray-500 mt-1">Model Confidence: {{ results.mlPrediction?.model_confidence }}%</p>
          </div>
        </div>
      </div>

      <!-- Image Comparison -->
      <div class="bg-white rounded-lg shadow-lg p-6">
        <h3 class="text-xl font-bold text-gray-800 mb-4">Image Analysis</h3>
        <div class="grid md:grid-cols-2 gap-4">
          <div>
            <h4 class="font-medium text-gray-700 mb-2">Original Check</h4>
            <img [src]="getImageUrl('original')" alt="Original check" class="w-full border rounded shadow-sm" (error)="onImageError($event)" />
          </div>
          <div>
            <h4 class="font-medium text-gray-700 mb-2">Annotated (Suspicious Regions)</h4>
            <img [src]="getImageUrl('annotated')" alt="Annotated check" class="w-full border rounded shadow-sm" (error)="onImageError($event)" />
          </div>
        </div>
      </div>

      <!-- Stage History -->
      <div class="bg-white rounded-lg shadow-lg p-6">
        <h3 class="text-xl font-bold text-gray-800 mb-4">Processing Timeline</h3>
        <div class="space-y-3">
          <div *ngFor="let stage of results.stageHistory" class="flex items-center space-x-3 p-3 bg-gray-50 rounded">
            <div class="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center">
              <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
              </svg>
            </div>
            <div class="flex-1">
              <p class="font-medium text-gray-800">{{ stage.stageName }}</p>
              <p class="text-xs text-gray-600">{{ formatTimestamp(stage.completedAt) }}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [],
})
export class ResultsComponent {
  private apiService = inject(ApiService);

  @Input() results!: AnalysisResults;
  @Input() jobId!: string;
  @Output() newAnalysis = new EventEmitter<void>();

  getImageUrl(type: 'original' | 'annotated'): string {
    return this.apiService.getImageUrl(this.jobId, type);
  }

  onImageError(event: any) {
    event.target.src =
      'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200"><rect width="400" height="200" fill="%23f3f4f6"/><text x="50%" y="50%" text-anchor="middle" fill="%239ca3af">Image not available</text></svg>';
  }

  getVerdictDescription(verdict: string): string {
    switch (verdict) {
      case 'LEGITIMATE':
        return 'Check appears to be authentic';
      case 'SUSPICIOUS':
        return 'Manual review recommended';
      case 'FRAUDULENT':
        return 'High risk of fraud detected';
      default:
        return '';
    }
  }

  formatAmount(amount?: number): string {
    if (!amount) return 'N/A';
    return '$' + amount.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  formatTimestamp(timestamp: string): string {
    try {
      const date = new Date(timestamp);
      return date.toLocaleString();
    } catch {
      return timestamp;
    }
  }
}
