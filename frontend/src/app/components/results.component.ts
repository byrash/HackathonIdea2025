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

      <!-- GPS Location (if available) -->
      <div *ngIf="results.metadata_analysis?.gps_location?.gps_found" class="bg-white rounded-lg shadow-lg p-6">
        <h3 class="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <span class="mr-2">📍</span>
          GPS Location
        </h3>
        <div class="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
          <div class="flex-1">
            <div class="space-y-3">
              <div>
                <p class="text-sm text-gray-600">Location</p>
                <p class="text-lg font-medium text-gray-800">
                  {{ results.metadata_analysis.gps_location.city || 'Unknown' }}<span *ngIf="results.metadata_analysis.gps_location.state">, {{ results.metadata_analysis.gps_location.state }}</span>
                </p>
                <p class="text-sm text-gray-600">{{ results.metadata_analysis.gps_location.country }}</p>
              </div>
              <div>
                <p class="text-sm text-gray-600">Coordinates</p>
                <p class="font-mono text-sm text-gray-700">
                  {{ results.metadata_analysis.gps_location.latitude?.toFixed(6) }}°, {{ results.metadata_analysis.gps_location.longitude?.toFixed(6) }}°
                </p>
              </div>
              <div *ngIf="isUSALocation(results.metadata_analysis.gps_location)" 
                   class="inline-flex items-center px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">
                <svg class="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                </svg>
                USA Location (Lower Fraud Risk)
              </div>
            </div>
          </div>
          <div class="flex-shrink-0">
            <a 
              *ngIf="results.metadata_analysis.gps_location.map_url" 
              [href]="results.metadata_analysis.gps_location.map_url" 
              target="_blank"
              class="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors shadow-md"
            >
              <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"></path>
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"></path>
              </svg>
              View on Map
            </a>
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

      <!-- Technical Details (Expandable) -->
      <div class="bg-white rounded-lg shadow-lg p-6">
        <button 
          (click)="showTechnicalDetails = !showTechnicalDetails"
          class="w-full flex items-center justify-between text-left group"
        >
          <h3 class="text-xl font-bold text-gray-800 flex items-center">
            <span class="mr-2">🔧</span>
            Technical Details & Raw Data
          </h3>
          <svg 
            class="w-6 h-6 text-gray-600 transform transition-transform"
            [class.rotate-180]="showTechnicalDetails"
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
          </svg>
        </button>

        <div *ngIf="showTechnicalDetails" class="mt-6 space-y-6">
          <!-- EXIF Metadata -->
          <div class="border-t pt-4">
            <h4 class="font-semibold text-gray-700 mb-3 flex items-center">
              📸 EXIF Metadata ({{ getExifFieldCount() }} fields)
            </h4>
            <div *ngIf="results.metadata_analysis?.has_metadata" class="bg-gray-50 rounded-lg p-4">
              <!-- Camera Info -->
              <div *ngIf="results.metadata_analysis?.camera_analysis?.device_found" class="mb-4 p-3 bg-white rounded border">
                <p class="text-xs font-semibold text-gray-600 mb-2">Device Information</p>
                <div class="text-sm space-y-1">
                  <p *ngIf="results.metadata_analysis?.camera_analysis?.camera_make">
                    <span class="font-medium">Make:</span> {{ results.metadata_analysis.camera_analysis.camera_make }}
                  </p>
                  <p *ngIf="results.metadata_analysis?.camera_analysis?.camera_model">
                    <span class="font-medium">Model:</span> {{ results.metadata_analysis.camera_analysis.camera_model }}
                  </p>
                  <p *ngIf="results.metadata_analysis?.camera_analysis?.software">
                    <span class="font-medium">Software:</span> 
                    <span [class.text-red-600]="results.metadata_analysis.camera_analysis.software.toLowerCase().includes('photoshop')">
                      {{ results.metadata_analysis.camera_analysis.software }}
                    </span>
                  </p>
                  <p *ngIf="results.metadata_analysis?.camera_analysis?.is_scanner" class="text-blue-600">
                    ✓ Scanner device detected
                  </p>
                  <p *ngIf="results.metadata_analysis?.camera_analysis?.is_mobile" class="text-blue-600">
                    📱 Mobile device detected
                  </p>
                </div>
              </div>

              <!-- Date Information -->
              <div *ngIf="results.metadata_analysis?.creation_date_analysis?.date_found" class="mb-4 p-3 bg-white rounded border">
                <p class="text-xs font-semibold text-gray-600 mb-2">Date & Time</p>
                <div class="text-sm space-y-1">
                  <p *ngIf="results.metadata_analysis?.creation_date_analysis?.creation_date">
                    <span class="font-medium">Created:</span> {{ results.metadata_analysis.creation_date_analysis.creation_date }}
                  </p>
                  <p *ngIf="results.metadata_analysis?.creation_date_analysis?.modified_date">
                    <span class="font-medium">Modified:</span> {{ results.metadata_analysis.creation_date_analysis.modified_date }}
                  </p>
                  <div *ngIf="results.metadata_analysis?.creation_date_analysis?.inconsistencies?.length > 0" class="mt-2">
                    <p *ngFor="let issue of results.metadata_analysis.creation_date_analysis.inconsistencies" class="text-red-600 text-xs">
                      ⚠️ {{ issue }}
                    </p>
                  </div>
                </div>
              </div>

              <!-- GPS Location Information -->
              <div *ngIf="results.metadata_analysis?.gps_location" class="mb-4 p-3 bg-white rounded border">
                <p class="text-xs font-semibold text-gray-600 mb-2">📍 GPS Location</p>
                <div *ngIf="results.metadata_analysis.gps_location.gps_found" class="text-sm space-y-2">
                  <div>
                    <p class="font-medium text-gray-800">
                      {{ results.metadata_analysis.gps_location.city || 'Unknown' }}<span *ngIf="results.metadata_analysis.gps_location.state">, {{ results.metadata_analysis.gps_location.state }}</span>
                    </p>
                    <p class="text-xs text-gray-600">{{ results.metadata_analysis.gps_location.country }}</p>
                  </div>
                  <div class="text-xs text-gray-600 font-mono">
                    {{ results.metadata_analysis.gps_location.latitude?.toFixed(6) }}°, {{ results.metadata_analysis.gps_location.longitude?.toFixed(6) }}°
                  </div>
                  <a 
                    *ngIf="results.metadata_analysis.gps_location.map_url" 
                    [href]="results.metadata_analysis.gps_location.map_url" 
                    target="_blank"
                    class="inline-flex items-center px-3 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700 transition-colors"
                  >
                    🗺️ View on Map
                  </a>
                  <div *ngIf="isUSALocation(results.metadata_analysis.gps_location)" class="text-green-600 text-xs mt-1">
                    ✓ USA location (lower fraud risk)
                  </div>
                </div>
                <div *ngIf="!results.metadata_analysis.gps_location.gps_found" class="text-xs text-gray-500">
                  No GPS data found (normal for scanned checks)
                </div>
              </div>

              <!-- All EXIF Fields (Collapsible) -->
              <div class="mb-4">
                <button 
                  (click)="showAllExif = !showAllExif"
                  class="text-sm text-blue-600 hover:text-blue-700 flex items-center"
                >
                  <span>{{ showAllExif ? '▼' : '▶' }}</span>
                  <span class="ml-1">Show all {{ getExifFieldCount() }} EXIF fields</span>
                </button>
                <div *ngIf="showAllExif" class="mt-3 p-3 bg-white rounded border max-h-64 overflow-y-auto">
                  <div *ngFor="let field of getExifFields()" class="text-xs py-1 border-b last:border-b-0">
                    <span class="font-mono text-gray-600">{{ field.key }}:</span>
                    <span class="ml-2 text-gray-800">{{ field.value }}</span>
                  </div>
                </div>
              </div>

              <!-- Metadata Warnings -->
              <div *ngIf="results.metadata_analysis?.flags?.length > 0" class="p-3 bg-red-50 rounded border border-red-200">
                <p class="text-xs font-semibold text-red-700 mb-2">⚠️ Metadata Warnings</p>
                <ul class="text-sm space-y-1">
                  <li *ngFor="let flag of results.metadata_analysis.flags" class="text-red-600">• {{ flag }}</li>
                </ul>
              </div>
            </div>
            <div *ngIf="!results.metadata_analysis?.has_metadata" class="bg-red-50 rounded-lg p-4 border border-red-200">
              <p class="text-sm text-red-700">⚠️ No EXIF metadata found - possibly stripped (suspicious)</p>
            </div>
          </div>

          <!-- ELA (Error Level Analysis) -->
          <div class="border-t pt-4">
            <h4 class="font-semibold text-gray-700 mb-3 flex items-center">
              🔬 ELA - Error Level Analysis
            </h4>
            <div class="bg-gray-50 rounded-lg p-4">
              <p class="text-sm text-gray-600 mb-3">
                ELA detects digital alterations by analyzing JPEG compression inconsistencies.
                Edited regions show different compression levels than the rest of the image.
              </p>
              <div *ngIf="results.forensicAnalysis?.errorLevelAnalysis" class="space-y-3">
                <div class="grid grid-cols-2 gap-4">
                  <div class="p-3 bg-white rounded border">
                    <p class="text-xs text-gray-600">Suspicious Regions</p>
                    <p class="text-2xl font-bold" [class.text-red-600]="results.forensicAnalysis.errorLevelAnalysis.suspicious_regions?.length > 0">
                      {{ results.forensicAnalysis.errorLevelAnalysis.suspicious_regions?.length || 0 }}
                    </p>
                  </div>
                  <div class="p-3 bg-white rounded border">
                    <p class="text-xs text-gray-600">Confidence Level</p>
                    <p class="text-2xl font-bold text-gray-800">
                      {{ results.forensicAnalysis.errorLevelAnalysis.confidence || 0 }}%
                    </p>
                  </div>
                </div>

                <!-- Suspicious Regions Details -->
                <div *ngIf="results.forensicAnalysis.errorLevelAnalysis.suspicious_regions?.length > 0" class="mt-3">
                  <p class="text-xs font-semibold text-gray-700 mb-2">Detected Regions:</p>
                  <div *ngFor="let region of results.forensicAnalysis.errorLevelAnalysis.suspicious_regions; let i = index" 
                       class="p-2 bg-white rounded border mb-2">
                    <div class="flex justify-between items-start">
                      <div class="text-xs">
                        <p class="font-semibold">Region #{{ i + 1 }}</p>
                        <p class="text-gray-600">
                          Position: ({{ region.coordinates?.x }}, {{ region.coordinates?.y }})
                          Size: {{ region.coordinates?.width }}x{{ region.coordinates?.height }}px
                        </p>
                      </div>
                      <span 
                        class="px-2 py-1 rounded text-xs font-semibold"
                        [class.bg-red-100]="region.severity === 'HIGH'"
                        [class.text-red-700]="region.severity === 'HIGH'"
                        [class.bg-yellow-100]="region.severity === 'MEDIUM'"
                        [class.text-yellow-700]="region.severity === 'MEDIUM'"
                      >
                        {{ region.severity }}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Clone Detection -->
          <div class="border-t pt-4" *ngIf="results.forensicAnalysis?.cloneDetection">
            <h4 class="font-semibold text-gray-700 mb-3 flex items-center">
              👯 Clone Detection
            </h4>
            <div class="bg-gray-50 rounded-lg p-4">
              <p class="text-sm text-gray-600 mb-3">
                Detects copy-pasted regions (e.g., duplicated signatures or amounts).
              </p>
              <div class="p-3 bg-white rounded border">
                <p class="text-sm">
                  <span class="font-medium">Status:</span>
                  <span [class.text-red-600]="results.forensicAnalysis.cloneDetection.duplicates_found"
                        [class.text-green-600]="!results.forensicAnalysis.cloneDetection.duplicates_found">
                    {{ results.forensicAnalysis.cloneDetection.duplicates_found ? '⚠️ Duplicates Found' : '✓ No Duplicates' }}
                  </span>
                </p>
                <p class="text-xs text-gray-600 mt-1">
                  Duplicate Regions: {{ results.forensicAnalysis.cloneDetection.duplicate_count || 0 }}
                </p>
              </div>
            </div>
          </div>

          <!-- Edge Analysis -->
          <div class="border-t pt-4" *ngIf="results.forensicAnalysis?.edgeAnalysis">
            <h4 class="font-semibold text-gray-700 mb-3 flex items-center">
              📐 Edge Analysis
            </h4>
            <div class="bg-gray-50 rounded-lg p-4">
              <p class="text-sm text-gray-600 mb-3">
                Analyzes edges and boundaries for suspicious straight lines or cut-and-paste artifacts.
              </p>
              <div class="p-3 bg-white rounded border">
                <p class="text-sm">
                  <span class="font-medium">Irregular Edges Found:</span>
                  <span [class.text-red-600]="results.forensicAnalysis.edgeAnalysis.irregular_edges?.length > 5"
                        [class.text-yellow-600]="results.forensicAnalysis.edgeAnalysis.irregular_edges?.length > 0 && results.forensicAnalysis.edgeAnalysis.irregular_edges?.length <= 5">
                    {{ results.forensicAnalysis.edgeAnalysis.irregular_edges?.length || 0 }}
                  </span>
                </p>
                <p class="text-xs text-gray-600 mt-1" *ngIf="results.forensicAnalysis.edgeAnalysis.irregular_edges?.length > 5">
                  ⚠️ High number of irregular edges detected - possible tampering
                </p>
              </div>
            </div>
          </div>

          <!-- Raw JSON Data -->
          <div class="border-t pt-4">
            <button 
              (click)="showRawJson = !showRawJson"
              class="text-sm text-blue-600 hover:text-blue-700 flex items-center"
            >
              <span>{{ showRawJson ? '▼' : '▶' }}</span>
              <span class="ml-1">Show raw JSON data</span>
            </button>
            <div *ngIf="showRawJson" class="mt-3 bg-gray-900 rounded-lg p-4 overflow-x-auto">
              <pre class="text-xs text-green-400 font-mono">{{ results | json }}</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [],
})
export class ResultsComponent {
  private readonly apiService = inject(ApiService);

  @Input() results!: AnalysisResults;
  @Input() jobId!: string;
  @Output() newAnalysis = new EventEmitter<void>();

  showTechnicalDetails = false;
  showAllExif = false;
  showRawJson = false;

  getImageUrl(type: 'original' | 'annotated'): string {
    return this.apiService.getImageUrl(this.jobId, type);
  }

  getExifFieldCount(): number {
    return this.results.metadata_analysis?.exif_field_count || 0;
  }

  getExifFields(): Array<{key: string, value: string}> {
    const exifData = this.results.metadata_analysis?.all_exif_data || {};
    return Object.entries(exifData).map(([key, value]) => ({ 
      key, 
      value: typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value ?? '')
    }));
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

  isUSALocation(gpsLocation: any): boolean {
    if (!gpsLocation?.country) return false;
    const country = gpsLocation.country.toLowerCase();
    return country.includes('united states') || country.includes('usa') || country === 'us';
  }
}
