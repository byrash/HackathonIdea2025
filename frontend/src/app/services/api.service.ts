import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface UploadResponse {
  jobId: string;
  status: string;
  message: string;
}

export interface ProgressUpdate {
  jobId: string;
  status: string;
  currentStage: number;
  currentPercentage: number;
  message: string;
  timestamp: string;
}

export interface AnalysisResults {
  jobId: string;
  status: string;
  overallRiskScore: number;
  verdict: 'LEGITIMATE' | 'SUSPICIOUS' | 'FRAUDULENT';
  extractedData: {
    checkNumber?: string;
    date?: string;
    payee?: string;
    amountNumeric?: number;
    amountWritten?: string;
    routingNumber?: string;
    accountNumber?: string;
  };
  validationResults: any;
  forensicAnalysis: any;
  securityFeatures: any;
  mlPrediction: any;
  recommendations: string[];
  metadata_analysis: any;
  stageHistory: any[];
}

@Injectable({
  providedIn: 'root',
})
export class ApiService {
  private http = inject(HttpClient);
  private apiUrl = 'http://localhost:8000/api';

  /**
   * Upload check image for analysis
   */
  uploadCheck(file: File): Observable<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<UploadResponse>(`${this.apiUrl}/checks/upload`, formData);
  }

  /**
   * Listen to progress updates via Server-Sent Events (SSE)
   */
  listenToProgress(jobId: string, onProgress: (update: ProgressUpdate) => void, onComplete: () => void, onError: (error: any) => void): EventSource {
    const eventSource = new EventSource(`${this.apiUrl}/checks/${jobId}/progress`);

    eventSource.onmessage = (event) => {
      try {
        const data: ProgressUpdate = JSON.parse(event.data);
        onProgress(data);

        if (data.status === 'COMPLETED' || data.status === 'FAILED') {
          eventSource.close();
          onComplete();
        }
      } catch (error) {
        console.error('Error parsing SSE data:', error);
        onError(error);
      }
    };

    eventSource.onerror = (error) => {
      console.error('SSE Error:', error);
      eventSource.close();
      onError(error);
    };

    return eventSource;
  }

  /**
   * Get complete analysis results
   */
  getResults(jobId: string): Observable<AnalysisResults> {
    return this.http.get<AnalysisResults>(`${this.apiUrl}/checks/${jobId}/results`);
  }

  /**
   * Get check image URL
   */
  getImageUrl(jobId: string, type: 'original' | 'annotated'): string {
    return `${this.apiUrl}/checks/${jobId}/image/${type}`;
  }
}
